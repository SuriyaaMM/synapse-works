import { createEventDispatcher } from 'svelte';
import type { FlowNode, FlowEdge, GraphState } from './types';
import { layerTypes } from './types';
import { convertLayerToConfig } from './layer';

class GraphStore {
  private state = $state<GraphState>({
    nodes: [],
    edges: [],
    selectedNode: null,
    selectedEdge: null,
    selectedLayerType: null,
    nodeCounter: 1,
    buildResult: null,
    graphValidationResult: null,
    loading: false,
    error: null
  });

  // Getters
  get nodes() { return this.state.nodes; }
  get edges() { return this.state.edges; }
  get selectedNode() { return this.state.selectedNode; }
  get selectedEdge() { return this.state.selectedEdge; }
  get selectedLayerType() { return this.state.selectedLayerType; }
  get nodeCounter() { return this.state.nodeCounter; }
  get buildResult() { return this.state.buildResult; }
  get graphValidationResult() { return this.state.graphValidationResult; }
  get loading() { return this.state.loading; }
  get error() { return this.state.error; }

  // Setters
  setNodes(nodes: FlowNode[]) { this.state.nodes = nodes; }
  setEdges(edges: FlowEdge[]) { this.state.edges = edges; }
  setSelectedNode(node: FlowNode | null) { this.state.selectedNode = node; }
  setSelectedEdge(edge: FlowEdge | null) { this.state.selectedEdge = edge; }
  setSelectedLayerType(layerType: typeof layerTypes[number] | null) { this.state.selectedLayerType = layerType; }
  setNodeCounter(counter: number) { this.state.nodeCounter = counter; }
  setBuildResult(result: any) { this.state.buildResult = result; }
  setGraphValidationResult(result: any) { this.state.graphValidationResult = result; }
  setLoading(loading: boolean) { this.state.loading = loading; }
  setError(error: string | null) { this.state.error = error; }

  // Actions
  incrementNodeCounter() { this.state.nodeCounter++; }
  
  clearSelection() {
    this.state.selectedNode = null;
    this.state.selectedEdge = null;
    this.state.selectedLayerType = null;
    this.state.error = null;
  }

  updateNodePosition(nodeId: string, position: { x: number; y: number }) {
    const nodeIndex = this.state.nodes.findIndex(n => n.id === nodeId);
    if (nodeIndex !== -1) {
      this.state.nodes[nodeIndex].position = position;
    }
  }

  updateNodesFromGraphData(graphData: any) {
    if (graphData.layers) {
      const existingPositions = new Map<string, { x: number; y: number }>();
      console.log(existingPositions)
      this.state.nodes.forEach(node => {
        existingPositions.set(node.id, node.position);
      });

      const horizontalSpacing = 100;
      const verticalSpacing = 100;
      const startX = 100;
      const startY = 100;

      this.state.nodes = graphData.layers.map((layer: any, index: number) => {
        const existingNode = this.state.nodes.find(n => n.id === layer.id);
        const layerType = layerTypes.find(lt => lt.type === layer.type);
        
        let position: { x: number; y: number };
        if (existingPositions.has(layer.id)) {
          position = existingPositions.get(layer.id)!;
        } else {
          const row = index;
          let xOffset;
          
          if (row % 2 === 0) {
            xOffset = (row / 2) * horizontalSpacing;
          } else {
            xOffset = ((row - 1) / 2 + 1) * horizontalSpacing + 300;
          }
          
          position = {
            x: startX + xOffset,
            y: startY + (row * verticalSpacing)
          };
        }
        
        return {
          id: layer.id,
          selectable: true,
          position,
          data: {
            label: `${layer.name || layer.id}[${layer.type}]`,
            layerType: layer.type,
            layerConfig: convertLayerToConfig(layer),
            color: layerType?.color || '#374151'
          },
          selected: existingNode?.selected || false
        };
      });
    }

    if (graphData.edges) {
      this.state.edges = [];
      graphData.edges.forEach((edge: any) => {
        if (edge.target_ids && Array.isArray(edge.target_ids)) {
          edge.target_ids.forEach((targetId: string) => {
            this.state.edges.push({
              id: `${edge.source_id}-${targetId}`,
              source: edge.source_id,
              target: targetId
            });
          });
        }
      });
    }
  }

  getStateForStorage() {
    return {
      nodes: this.state.nodes,
      edges: this.state.edges,
      buildResult: this.state.buildResult,
      nodeCounter: this.state.nodeCounter
    };
  }

  loadStateFromStorage(savedState: any) {
    this.state.nodes = savedState.nodes || [];
    this.state.edges = savedState.edges || [];
    this.state.buildResult = savedState.buildResult || null;
    this.state.nodeCounter = savedState.nodeCounter || 1;
  }
}

export const graphStore = new GraphStore();