from collections import defaultdict

def modify_objects(nodes, edges):
    new_nodes = defaultdict(dict)
    for node in nodes:
        node_type = node.get('type', 'base')
        if node_type == 'base':
            new_nodes.update({
                    int(node['id']): {
                        'name': node['data'].pop('label'), 
                        'args': node['data']['Args'] if len(node['data']['Args']) > 0 else {}
                    }
                }
            )
        elif len(node['data']['include']['nodes']) > 1:
            new_nodes.update({
                    int(node['id']): {
                        'name': 'custom',
                        'label': f'# Custom node "{node["data"].pop("label")}"',
                        'nodes': node['data']['include']['nodes'],
                        'edges': node['data']['include']['edges']
                    }
                }
            )
    
    new_edges = defaultdict(dict)
    for edge in edges:
        new_edges.update({
                int(edge['source']): {
                    'id': edge['id'], 
                    'target': int(edge['target'])
                }
            }
        )
    return new_nodes, new_edges


def end_of_layer_block(layers, sequence, modules) -> list:
    if len(modules) <= 0:
        return []
    
    tabs = ',\n\t\t\t'
    layer = """
		self.layer_{} = nn.Sequential(
			{}
		)
    """
    layers.append(layer.format(len(layers) + 1, tabs.join(modules)))
    sequence.append(f"data = self.layer_{len(layers)}(data)")
    return []


def code_recursion(nodes, edges, layers, sequence, rec: int = 0, custom: str = ''):
    if rec == 1:
        sequence.append(f'{custom}')
    nodes, edges = modify_objects(nodes, edges)
    
    curr_node = 0
    modules = []
    while len(edges) > 0:
        curr_node = edges.pop(curr_node)['target']
        if nodes[curr_node]['name'] != 'custom':
            args = ', '.join(f'{key}={value}' for key, value in nodes[curr_node]['args'].items() if not isinstance(value, dict))
            if not nodes[curr_node]['name'].islower():
                modules.append(f"nn.{nodes[curr_node]['name']}({args})")
            else:
                modules = end_of_layer_block(layers, sequence, modules)
                sequence.append(f"data = F.{nodes[curr_node]['name']}(data{(', ' if len(args) > 0 else '') + args})")
        else:
            modules = end_of_layer_block(layers, sequence, modules)
            code_recursion(nodes[curr_node]['nodes'], nodes[curr_node]['edges'], layers, sequence, rec + 1, nodes[curr_node]['label'])
    
    modules = end_of_layer_block(layers, sequence, modules)
    if rec == 1:
        sequence.append('')