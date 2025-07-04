import { createEventDispatcher } from 'svelte';
import type { FlowNode, FlowEdge, GraphState } from './types';
import { layerTypes } from './types';
import { convertLayerToConfig } from './layer';

class GraphStore {
  private initialState: GraphState = {
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
  };

  private state = $state<GraphState>({ ...this.initialState });

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

  resetToInitialState() {
    console.log('Resetting graph store to initial state');
    
    // Reset all state properties to their initial values
    this.state.nodes = [];
    this.state.edges = [];
    this.state.selectedNode = null;
    this.state.selectedEdge = null;
    this.state.selectedLayerType = null;
    this.state.nodeCounter = 1;
    this.state.buildResult = null;
    this.state.graphValidationResult = null;
    this.state.loading = false;
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
    const existingNodes = new Map(this.state.nodes.map(node => [node.id, node]));
    const newLayerIds = new Set(graphData.layers.map((layer: any) => layer.id));
    
    // Remove nodes that are no longer in the graph
    this.state.nodes = this.state.nodes.filter(node => newLayerIds.has(node.id));
    
    // Process each layer
    graphData.layers.forEach((layer: any, index: number) => {
      const existingNode = existingNodes.get(layer.id);
      const layerType = layerTypes.find(lt => lt.type === layer.type);
      
      if (existingNode) {
        // Update existing node - keep all existing data, just update what might have changed
        const nodeIndex = this.state.nodes.findIndex(n => n.id === layer.id);
        if (nodeIndex !== -1) {
          this.state.nodes[nodeIndex] = {
            ...existingNode,
            data: {
              ...existingNode.data,
              label: `${layer.name || layer.id}[${layer.type}]`,
            }
          };
        }
      } else {
        // Add new node
        const horizontalSpacing = 100;
        const verticalSpacing = 100;
        const startX = 100;
        const startY = 100;
        
        // Position new nodes
        let position: { x: number; y: number };
        
        if (this.state.nodes.length > 0) {
          // Find bounds of existing nodes
          const positions = this.state.nodes.map(n => n.position);
          const maxX = Math.max(...positions.map(p => p.x));
          const minY = Math.min(...positions.map(p => p.y));
          
          // Position new node to the right of existing nodes
          position = {
            x: maxX + horizontalSpacing * 2,
            y: minY
          };
        } else {
          position = { x: startX, y: startY };
        }
        
        const newNode = {
          id: layer.id,
          selectable: true,
          position,
          data: {
            label: `${layer.name || layer.id}[${layer.type}]`,
            layerType: layer.type,
            layerConfig: convertLayerToConfig(layer),
            color: layerType?.color || '#374151'
          },
          selected: false
        };
        
        this.state.nodes.push(newNode);
      }
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
    console.log('Loading state from storage:', savedState);
    
    this.state.nodes = savedState.nodes || [];
    this.state.edges = savedState.edges || [];
    this.state.buildResult = savedState.buildResult || null;
    this.state.nodeCounter = savedState.nodeCounter || 1;
    
    // Clear selections when loading from storage
    this.clearSelection();
  }

  // Helper method to check if the graph is empty
  isEmpty(): boolean {
    return this.state.nodes.length === 0 && this.state.edges.length === 0;
  }

  // Helper method to get a summary of the current state
  getStateSummary(): string {
    return `Nodes: ${this.state.nodes.length}, Edges: ${this.state.edges.length}, Counter: ${this.state.nodeCounter}`;
  }
}

export const graphStore = new GraphStore();