from config import logging

import torch
from torch import nn
from torch.fx import Graph, Node

from typedefs import * 
from typing import List, Any
from backendUtils import parseFromLayerConfig 
from backendTorchUtils import torch_layer_name_map

def parseModuleGraphInput(module_graph_dict: Dict[str, Any]) -> ModuleGraph:
    # this layers list will never be empty
    raw_layers: List[TSLayerInput] = module_graph_dict["layers"]
    parsed_layers: List[LayerConfig] = []
    # populate parsed_layers which
    for raw_layer in raw_layers:
        parsed_layers.append(parseFromLayerConfig(raw_layer))

    # get raw edges
    parsed_edges: List[ModuleAdjacencyList] = []
    # edges will never be empty
    for edge_dict in module_graph_dict["edges"]:
        parsed_edges.append(ModuleAdjacencyList(
            source_id=edge_dict["source_id"],
            target_ids=edge_dict["target_ids"]
        ))
    
    # topologically sorted layer ids
    sorted_ids: List[str] = module_graph_dict["sorted"]
    
    # create and return this module graph for building a graph
    return ModuleGraph(
        layers=parsed_layers,
        edges=parsed_edges,
        sorted_ids=sorted_ids
    )

def buildModuleGraph(module_graph: ModuleGraph) -> nn.Module:
    
    graph = Graph()
    # create an map <id, LayerConfig> 
    layer_map: Dict[str, LayerConfig] = {layer['id']: layer for layer in module_graph.layers}
    # create another map for <id, Output Nodes>
    layer_output_nodes: Dict[str, Node] = {}
    # predecessor map <id, list of input ids that needs to be processed before we can process this layer>
    predecessor_map: Dict[str, list[str]] = {layer['id'] : [] for layer in module_graph.layers}
    for edge in module_graph.edges:
        for target_id in edge.target_ids:
            predecessor_map[target_id].append(edge.source_id)
    # graph input node (initially is the input we receive)
    graph_input_node: Node = None # Initialize to None # type:ignore
    # create another map for <id, Module> 
    nn_modules: Dict[str, nn.Module] = {}
    # we will go by topologically sorted id's 
    for sorted_id in module_graph.sorted_ids:
        layer_config = layer_map[sorted_id]
        layer_id = layer_config["id"]
        # get the corresponding nn.Module for this layer and store it in the map
        nn_modules[layer_id] = torch_layer_name_map(layer_config["type"])(**layer_config["kwargs"])
        # initialize graph input node with generic input like x
        if graph_input_node is None:
            graph_input_node = graph.placeholder("x") 

        current_layer_inputs = []
        # get the list of layer's input's this layer depends on
        predecessors_ids = predecessor_map[layer_id]

        if not predecessors_ids: 
            # if there are no predecessor's its input comes from graph_input_node
            if graph_input_node not in current_layer_inputs:
                current_layer_inputs.append(graph_input_node)
        else:
            # simulate output by passing through all the predecessors for this layer
            for predecessor_id in predecessors_ids:
                if predecessor_id in layer_output_nodes:
                    current_layer_inputs.append(layer_output_nodes[predecessor_id])
        
        # if input is only one meaninig standard torch module
        if len(current_layer_inputs) == 1:
            input_for_call = current_layer_inputs[0]
        # TODO(mms) Yet to implement and handle for concat like layers that require
        # multiple inputs
        elif len(current_layer_inputs) > 1:
            logging.info(f"Layer {sorted_id} has multiple inputs. Needs specific handling (e.g., concatenation).")
            input_for_call = tuple(current_layer_inputs)

        if isinstance(input_for_call, tuple):
            output_node = graph.call_module(layer_id, args=input_for_call)
        else:
            output_node = graph.call_module(layer_id, args=(input_for_call,))
        # store the output node
        layer_output_nodes[sorted_id] = output_node
    
    # get the final output  node
    final_output_node_id = module_graph.sorted_ids[-1]
    final_output_node = layer_output_nodes[final_output_node_id]
    # set the final output node
    graph.output(final_output_node)
    # construct the dynamic nn.Module class
    class DynamicModuleForFX(nn.Module):
        def __init__(self, named_modules: Dict[str, nn.Module]):
            super().__init__()
            for name, module_instance in named_modules.items():
                self.add_module(name, module_instance) 

    fx_module = torch.fx.GraphModule(DynamicModuleForFX(nn_modules), graph)
    
    logging.info("--- Generated FX GraphModule ---")
    logging.info(fx_module)
    logging.info("--------------------------------")

    return fx_module