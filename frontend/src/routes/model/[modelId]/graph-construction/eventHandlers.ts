import { graphStore } from './graphStore.svelte';
import type { FlowNode } from './types';

let lastClickTime = 0;
const DOUBLE_CLICK_DELAY = 300;

export function createEventHandlers(
  createConnection: (fromId: string, toId: string) => void,
  showNodeDetailsView: (node: FlowNode) => void,
  saveStateToStorage: () => void
) {
  
  function onNodeClick(event: any) {
    console.log('Node clicked:', event.detail?.node || event.node);
    const node = event.detail?.node || event.node;
    
    if (node) {
      const currentTime = Date.now();
      
      // Check for double-click
      if (currentTime - lastClickTime < DOUBLE_CLICK_DELAY && graphStore.selectedNode?.id === node.id) {
        showNodeDetailsView(node);
      } else {
        graphStore.setSelectedNode(node);
        graphStore.setSelectedLayerType(null);
        graphStore.setSelectedEdge(null);
      }
      
      lastClickTime = currentTime;
    }
  }

  function onEdgeClick(event: any) {
    console.log('Edge clicked:', event);
    event.stopPropagation?.();

    const edge = event.detail?.edge || event.edge;
    if (edge) {
      graphStore.setSelectedEdge({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type
      });
      graphStore.setSelectedNode(null);
      graphStore.setSelectedLayerType(null);
    }
  }

  function onConnect(event: any) {
    console.log('Connect event:', event.detail || event);
    const connection = event.detail || event;
    if (connection.source && connection.target) {
      createConnection(connection.source, connection.target);
    }
  }

  function onNodesChange(event: CustomEvent<any[]>) {
    console.log("onNodesChange triggered:", event.detail);
    const changes = event.detail;
    let hasPositionChanges = false;
    
    if (!changes || !Array.isArray(changes)) {
      console.log("No valid changes array");
      return;
    }
    
    changes.forEach((change, index) => {
      console.log(`Change ${index}:`, change);
      
      // Handle different types of changes
      if (change.type === 'position' && change.position && change.id) {
        console.log(`Updating position for node ${change.id}:`, change.position);
        graphStore.updateNodePosition(change.id, change.position);
        hasPositionChanges = true;
      }
      // Also handle 'select' type changes if they exist
      else if (change.type === 'select' && change.id) {
        console.log(`Selection change for node ${change.id}:`, change.selected);
        // Handle selection state if needed
      }
      // Handle dimension changes
      else if (change.type === 'dimensions' && change.id) {
        console.log(`Dimension change for node ${change.id}:`, change.dimensions);
        // Handle dimension updates if needed
      }
    });
    
    console.log("Has position changes:", hasPositionChanges);
    
    // Only save to storage if there were actual position changes
    if (hasPositionChanges) {
      console.log("Saving to storage due to position changes");
      saveStateToStorage();
    }
  }

  function onNodeDrag(event: { node: any, event: MouseEvent }) {
    const node = event.node;
    if (node && node.position) {
      graphStore.updateNodePosition(node.id, node.position);
    }
  }

  function onNodeDragStop(event: { event: MouseEvent; targetNode: any; nodes: any[] }) {
    const node = event.targetNode;
    if (node && node.position) {
      graphStore.updateNodePosition(node.id, node.position);
      saveStateToStorage();
    }
  }


  return {
    onNodeClick,
    onEdgeClick,
    onConnect,
    onNodesChange,
    onNodeDrag,
    onNodeDragStop
  };
}