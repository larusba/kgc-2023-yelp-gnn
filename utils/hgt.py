import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                # if h.get(srctype) is not None and h.get(dsttype) is not None and edge_dict.get((srctype, etype, dsttype)) is not None:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h['src'][srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h['src'][srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h['dst'][dsttype]).view(-1, self.n_heads, self.d_k)
                
                # print(self.edge_dict)
                # print((srctype, etype, dsttype))
                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                # print(q.size())
                # print(sub_graph)
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v
                # eids.append(e_id)

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))

                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)
            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items() if etype in G.canonical_etypes}, cross_reducer = 'mean')
            # il controllo in etypes serve per non far rompere il codice quando non abbiamo tutti i tipi nel subgraph
            
            new_h = {'src':{}, 'dst': {}}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                if G.dstdata['t'].get(ntype) is not None:
                    t = F.gelu(G.dstdata['t'][ntype]).view(-1, self.out_dim)
                    trans_out = self.drop(self.a_linears[n_id](t))
                    trans_out = trans_out * alpha + h['dst'][ntype] * (1-alpha)
                    if self.use_norm:
                        new_h['src'][ntype] = self.norms[n_id](trans_out)
                    else:
                        new_h['src'][ntype] = trans_out
                else:
                    print("t not found")
            return new_h

class HGT(nn.Module):
    def __init__(self, node_dict, edge_dict, features_dim_dict, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.features_dim_dict = features_dim_dict
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleDict()
        self.input_norm = nn.LayerNorm(n_hid)
        for ntype in self.features_dim_dict:
            self.adapt_ws.add_module(ntype, nn.Linear(features_dim_dict[ntype], n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out1 = nn.Linear(n_hid, n_hid//2)
        self.out2 = nn.Linear(n_hid//2, n_out)

    def forward(self, G, out_key):
        h = { 'src': {}, 'dst': {} }
        for ntype in G[0].srcdata['feat']:
           h['src'][ntype] = self.input_norm(F.gelu(self.adapt_ws[ntype](G[0].srcdata['feat'][ntype])))
        for i in range(self.n_layers):
            for ntype in G[i].dstdata['feat']:
                h['dst'][ntype] = self.input_norm(F.gelu(self.adapt_ws[ntype](G[i].dstdata['feat'][ntype])))
            h = self.gcs[i](G[i], h)
        if isinstance(out_key, str):
            h['src'][out_key] = torch.nn.functional.normalize(h['src'][out_key])
            return self.out2(F.gelu(self.out1(h['src'][out_key])))
        elif isinstance(out_key, tuple) and len(out_key)==2:
            h['src'][out_key[0]] = torch.nn.functional.normalize(h['src'][out_key[0]])
            h['src'][out_key[1]] = torch.nn.functional.normalize(h['src'][out_key[1]])
            #controllare queste ultime righe quando si allena con blocks, dovrebbero non funzionare, capire perché
            G.nodes[out_key[0]].data['qh'] = self.out2(F.gelu(self.out1(h[out_key[0]])))
            G.nodes[out_key[1]].data['th'] = self.out2(F.gelu(self.out1(h[out_key[1]])))
            return G
