from graphviz import Digraph

#Manually create the GNN architecture visualization
def visualize_gnn():
    dot = Digraph(comment='GNN Architecture')

    # Add nodes
    dot.node('Input', 'Input Layer')

    # Graph Convolution Layers
    for i in range(6):
        dot.node(f'Conv{i + 1}', f'Graph Conv {i + 1}')
        dot.node(f'AggrModule{i + 1}', f'Aggr Module {i + 1}\n(avg_deg_lin, avg_deg_log)')
        dot.node(f'PreNN{i + 1}', f'Pre NN {i + 1}\n(weight, bias)')
        dot.node(f'PostNN{i + 1}', f'Post NN {i + 1}\n(weight, bias)')
        dot.node(f'Lin{i + 1}', f'Lin {i + 1}\n(weight, bias)')

        # Connect nodes within each convolution layer
        dot.edge(f'Conv{i + 1}', f'AggrModule{i + 1}')
        dot.edge(f'AggrModule{i + 1}', f'PreNN{i + 1}')
        dot.edge(f'PreNN{i + 1}', f'PostNN{i + 1}')
        dot.edge(f'PostNN{i + 1}', f'Lin{i + 1}')

        # Connect consecutive convolution layers
        if i > 0:
            dot.edge(f'Lin{i}', f'Conv{i + 1}')
        else:
            dot.edge('Input', f'Conv{i + 1}')

    # Feature Layers
    for i in range(6):
        dot.node(f'FeatureLayer{i + 1}',
                 f'Feature Layer {i + 1}\n(weight, bias, running_mean, running_var, num_batches_tracked)')

        # Connect Lin of last Conv layer to first feature layer
        if i == 0:
            dot.edge(f'Lin{6}', f'FeatureLayer{i + 1}')
        else:
            dot.edge(f'FeatureLayer{i}', f'FeatureLayer{i + 1}')

    # Head Layers
    for i in range(2):
        dot.node(f'HeadNN{i + 1}', f'Head NN {i + 1}\n(mlp weights and biases)')

        # Connect last feature layer to first head layer
        if i == 0:
            dot.edge(f'FeatureLayer{6}', f'HeadNN{i + 1}')
        else:
            dot.edge(f'HeadNN{i}', f'HeadNN{i + 1}')

    dot.node('Output', 'Output Layer')
    dot.edge('HeadNN2', 'Output')

    # Render and save the graph
    dot.render('gnn_schematic', format='png', view=True)



if __name__ == "__main__":
    visualize_gnn()