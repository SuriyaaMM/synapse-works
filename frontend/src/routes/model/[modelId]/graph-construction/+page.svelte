<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import client from '$lib/apolloClient';
  import { ADD_TO_GRAPH, CONNECT_NODES, DELETE_FROM_GRAPH, DISCONNECT_NODES, BUILD_MODULE_GRAPH } from '$lib/mutations';
  import type { LayerConfigInput, LinearLayerConfigInput, 
                Conv2dLayerConfigInput, Conv1dLayerConfigInput } from '../../../../../../source/types';
  import './graph-construction.css';

  // Type-safe dispatch functions
  const dispatch = createEventDispatcher<{
    layerAdded: { layerConfig: LayerConfigInput };
    layerDeleted: { layerId: string };
    connectionCreated: { fromId: string; toId: string };
    connectionDeleted: { connectionId: string };
  }>();

  // Available layer types
  const layerTypes = [
    { type: 'linear', color: '#DC2626', label: 'Linear' },
    { type: 'conv2d', color: '#4F46E5', label: 'Conv2D' },
    { type: 'conv1d', color: '#7C3AED', label: 'Conv1D' }
  ];
  
  // Types
  interface Node {
    id: string;
    type: string;
    x: number;
    y: number;
    color: string;
    layerConfig: LayerConfigInput;
    selected?: boolean;
  }

  interface Edge {
    id: string;
    from: string;
    to: string;
  }

  // Drag modes
  type DragMode = 'none' | 'move' | 'connect';

  // State
  let nodes: Node[] = [];
  let edges: Edge[] = [];
  let selectedLayerType: typeof layerTypes[number] | null = null;
  let selectedNode: Node | null = null;
  let selectedEdge: Edge | null = null;
  let nodeCounter = 1;
  let dragOffset = { x: 0, y: 0 };
  let dragMode: DragMode = 'none';
  let dragStartNode: Node | null = null;
  let canvas: HTMLDivElement | null = null;
  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let buildResult: any = null;
  
  // Connection preview state
  let showConnectionPreview = false;
  let connectionPreviewStart = { x: 0, y: 0 };
  let connectionPreviewEnd = { x: 0, y: 0 };
  
  // Form fields - Linear
  let layerName = '';
  let inFeatures = '';
  let outFeatures = '';
  let bias = true;
  
  // Form fields - Conv2d and Conv1d
  let inChannels = '';
  let outChannels = '';
  let kernelSize = '';
  let stride = '';
  let padding = '';
  let dilation = '';
  let groups = '1';
  let convBias = true;
  let paddingMode = 'zeros';

  // Extract modelId from URL
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    modelId = (modelIndex !== -1 && modelIndex + 1 < pathParts.length) ? pathParts[modelIndex + 1] : null;
  }

  $: if (modelId && typeof window !== 'undefined') {
    loadStateFromStorage();
  }
  
  function selectLayerType(layerType: typeof layerTypes[number]) {
    selectedLayerType = layerType;
    selectedNode = null;
    selectedEdge = null;
    
    resetFormFields();
  }

  function resetFormFields() {
    layerName = '';
    inFeatures = '';
    outFeatures = '';
    bias = true;
    inChannels = '';
    outChannels = '';
    kernelSize = '';
    stride = '';
    padding = '';
    dilation = '';
    groups = '1';
    convBias = true;
    paddingMode = 'zeros';
  }

  function parseArrayInput(input: string): number[] | null {
    if (!input.trim()) return null;
    
    const parts = input.split(',').map(s => s.trim());
    const numbers = parts.map(p => Number(p));
    
    if (numbers.some(n => isNaN(n) || n <= 0)) return null;
    
    return numbers;
  }

  function validateForm(): string | null {
    if (!selectedLayerType) return 'Layer type is required';
    
    if (selectedLayerType.type === 'linear') {
      const inFeaturesNum = Number(inFeatures);
      const outFeaturesNum = Number(outFeatures);
      
      if (!inFeatures || isNaN(inFeaturesNum) || inFeaturesNum <= 0) {
        return 'Input features must be a positive number';
      }
      if (!outFeatures || isNaN(outFeaturesNum) || outFeaturesNum <= 0) {
        return 'Output features must be a positive number';
      }
    } else if (selectedLayerType.type === 'conv2d') {
      const inChannelsNum = Number(inChannels);
      const outChannelsNum = Number(outChannels);
      const groupsNum = Number(groups);
      
      if (!inChannels || isNaN(inChannelsNum) || inChannelsNum <= 0) {
        return 'Input channels must be a positive number';
      }
      if (!outChannels || isNaN(outChannelsNum) || outChannelsNum <= 0) {
        return 'Output channels must be a positive number';
      }
      if (!kernelSize.trim()) {
        return 'Kernel size is required';
      }
      if (!groups || isNaN(groupsNum) || groupsNum <= 0) {
        return 'Groups must be a positive number';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(kernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
        return 'Kernel size must be an array of exactly two numbers (e.g., "3,3")';
      }
    } else if (selectedLayerType.type === 'conv1d') {
      const inChannelsNum = Number(inChannels);
      const outChannelsNum = Number(outChannels);
      const groupsNum = Number(groups);
      
      if (!inChannels || isNaN(inChannelsNum) || inChannelsNum <= 0) {
        return 'Input channels must be a positive number';
      }
      if (!outChannels || isNaN(outChannelsNum) || outChannelsNum <= 0) {
        return 'Output channels must be a positive number';
      }
      if (!kernelSize.trim()) {
        return 'Kernel size is required';
      }
      if (!groups || isNaN(groupsNum) || groupsNum <= 0) {
        return 'Groups must be a positive number';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(kernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "3")';
      }
    }
    
    return null;
  }

  function createLayerConfig(): LayerConfigInput {
    const layerNameValue = layerName.trim() || `${selectedLayerType!.type}Layer${nodeCounter}`;
    
    if (selectedLayerType!.type === 'linear') {
      const linearConfig: LinearLayerConfigInput = {
        name: layerNameValue,
        in_features: parseInt(inFeatures, 10),
        out_features: parseInt(outFeatures, 10),
        bias
      };
      
      return {
        type: 'linear',
        linear: linearConfig
      };
    } else if (selectedLayerType!.type === 'conv2d') {
      const conv2dConfig: Conv2dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3, 3],
        bias: convBias,
        padding_mode: paddingMode
      };
      
      // Add optional fields if provided
      if (stride.trim()) {
        const strideArray = parseArrayInput(stride);
        if (strideArray) {
          conv2dConfig.stride = strideArray;
        }
      }
      if (padding.trim()) {
        const paddingArray = parseArrayInput(padding);
        if (paddingArray) {
          conv2dConfig.padding = paddingArray;
        }
      }
      if (dilation.trim()) {
        const dilationArray = parseArrayInput(dilation);
        if (dilationArray) {
          conv2dConfig.dilation = dilationArray;
        }
      }
      if (groups.trim() && groups !== '1') {
        conv2dConfig.groups = [parseInt(groups, 10)];
      }
      
      return {
        type: 'conv2d',
        conv2d: conv2dConfig
      };
    } else if (selectedLayerType!.type === 'conv1d') {
      const conv1dConfig: Conv1dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3],
        bias: convBias,
        padding_mode: paddingMode
      };
      
      // Add optional fields if provided
      if (stride.trim()) {
        const strideArray = parseArrayInput(stride);
        if (strideArray) {
          conv1dConfig.stride = strideArray;
        }
      }
      if (padding.trim()) {
        const paddingArray = parseArrayInput(padding);
        if (paddingArray) {
          conv1dConfig.padding = paddingArray;
        }
      }
      if (dilation.trim()) {
        const dilationArray = parseArrayInput(dilation);
        if (dilationArray) {
          conv1dConfig.dilation = dilationArray;
        }
      }
      if (groups.trim() && groups !== '1') {
        conv1dConfig.groups = [parseInt(groups, 10)];
      }
      
      return {
        type: 'conv1d',
        conv1d: conv1dConfig
      };
    }

    throw new Error(`Unsupported layer type: ${selectedLayerType!.type}`);
  }
  
  async function addNodeToGraph() {
    const validationError = validateForm();
    if (validationError) {
      error = validationError;
      return;
    }

    if (!modelId) {
      error = 'Model ID is missing';
      return;
    }
    
    const layerConfig = createLayerConfig();
    loading = true;
    error = null;
    
    try {
      const response = await client.mutate({
        mutation: ADD_TO_GRAPH,
        variables: { layer_config: layerConfig },
        fetchPolicy: 'no-cache'
      });

      if (!response.data?.appendToModuleGraph) {
        throw new Error('Failed to add layer to graph - no data returned');
      }

      const graphData = response.data.appendToModuleGraph;
      
      updateNodesFromGraphData(graphData);
      
      nodeCounter++;
      saveStateToStorage();
      
      dispatch('layerAdded', { layerConfig });
      
      resetFormFields();

    } catch (err) {
      console.error('Error adding layer to graph:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Unknown error occurred while adding layer';
      }
    } finally {
      loading = false;
    }
  }

  function updateNodesFromGraphData(graphData: any) {
    if (graphData.layers) {
      nodes = graphData.layers.map((layer: any, index: number) => {
        const existingNode = nodes.find(n => n.id === layer.id);
        const layerType = layerTypes.find(lt => lt.type === layer.type);
        
        return {
          id: layer.id,
          type: layer.type,
          x: existingNode?.x || 300,
          y: existingNode?.y || 200 + (index * 80),
          color: layerType?.color || '#374151',
          layerConfig: convertLayerToConfig(layer),
          selected: existingNode?.selected || false
        };
      });
    }

    if (graphData.edges) {
      edges = [];
      graphData.edges.forEach((edge: any) => {
        if (edge.target_ids && Array.isArray(edge.target_ids)) {
          edge.target_ids.forEach((targetId: string) => {
            edges.push({
              id: `edge_${edge.source_id}_${targetId}`,
              from: edge.source_id,
              to: targetId
            });
          });
        }
      });
      edges = [...edges]
    }
  }

  function convertLayerToConfig(layer: any): LayerConfigInput {
    if (layer.type === 'linear') {
      return {
        type: 'linear',
        linear: {
          name: layer.name,
          in_features: layer.in_features,
          out_features: layer.out_features,
          bias: layer.bias ?? true
        }
      };
    } else if (layer.type === 'conv2d') {
      return {
        type: 'conv2d',
        conv2d: {
          name: layer.name,
          in_channels: layer.in_channels,
          out_channels: layer.out_channels,
          kernel_size: layer.kernel_size,
          padding: layer.padding,
          bias: layer.bias ?? true,
          padding_mode: layer.padding_mode || 'zeros'
        }
      };
    } else if (layer.type === 'conv1d') {
      return {
        type: 'conv1d',
        conv1d: {
          name: layer.name,
          in_channels: layer.in_channels,
          out_channels: layer.out_channels,
          kernel_size: layer.kernel_size,
          padding: layer.padding,
          bias: layer.bias ?? true,
          padding_mode: layer.padding_mode || 'zeros'
        }
      };
    }

    throw new Error(`Unsupported layer type: ${layer.type}`);
  }
  
  function selectNode(node: Node, event: MouseEvent) {
    event.stopPropagation();
    
    selectedNode = node;
    selectedLayerType = null;
    selectedEdge = null;
    populateFormFromNode(node);
  }

  async function createConnection(fromId: string, toId: string) {
    if (!modelId) {
      error = 'Model ID is missing';
      return;
    }

    loading = true;
    error = null;

    console.log(`Connecting layers: ${fromId} -> ${toId}`);

    try {
      const response = await client.mutate({
        mutation: CONNECT_NODES,
        variables: { 
          source_layer_id: fromId, 
          target_layer_id: toId 
        },
        fetchPolicy: 'no-cache'
      });

      console.log('Connect response:', response.data);

      if (!response.data?.connectInModuleGraph) {
        throw new Error('Failed to connect layers - no data returned');
      }

      const graphData = response.data.connectInModuleGraph;
      updateNodesFromGraphData(graphData);
      
      // Dispatch event for external listeners
      dispatch('connectionCreated', { fromId, toId });
      saveStateToStorage();

    } catch (err) {
      console.error('Error creating connection:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Unknown error occurred while connecting layers';
      }
    } finally {
      loading = false;
    }
  }

  function populateFormFromNode(node: Node) {
    const config = node.layerConfig;
    
    if (config.type === 'linear' && config.linear) {
      layerName = config.linear.name ?? '';
      inFeatures = config.linear.in_features.toString();
      outFeatures = config.linear.out_features.toString();
      bias = config.linear.bias ?? true;
    } else if (config.type === 'conv2d' && config.conv2d) {
      layerName = config.conv2d.name ?? '';
      inChannels = config.conv2d.in_channels.toString();
      outChannels = config.conv2d.out_channels.toString();
      kernelSize = config.conv2d.kernel_size.join(',');
      stride = config.conv2d.stride ? config.conv2d.stride.join(',') : '';
      padding = config.conv2d.padding ? config.conv2d.padding.join(',') : '';
      dilation = config.conv2d.dilation ? config.conv2d.dilation.join(',') : '';
      groups = config.conv2d.groups ? config.conv2d.groups[0].toString() : '1';
      convBias = config.conv2d.bias ?? true;
      paddingMode = config.conv2d.padding_mode ?? 'zeros';
    } else if (config.type === 'conv1d' && config.conv1d) {
      layerName = config.conv1d.name ?? '';
      inChannels = config.conv1d.in_channels.toString();
      outChannels = config.conv1d.out_channels.toString();
      kernelSize = config.conv1d.kernel_size.join(',');
      stride = config.conv1d.stride ? config.conv1d.stride.join(',') : '';
      padding = config.conv1d.padding ? config.conv1d.padding.join(',') : '';
      dilation = config.conv1d.dilation ? config.conv1d.dilation.join(',') : '';
      groups = config.conv1d.groups ? config.conv1d.groups[0].toString() : '1';
      convBias = config.conv1d.bias ?? true;
      paddingMode = config.conv1d.padding_mode ?? 'zeros';
    }
  }
  
  function selectEdge(edge: Edge, event: MouseEvent) {
    event.stopPropagation();
    selectedEdge = edge;
    selectedNode = null;
    selectedLayerType = null;
  }
  
  async function deleteSelectedNode() {
    if (!selectedNode || !modelId) return;
    
    loading = true;
    error = null;

    console.log(`Deleting layer: ${selectedNode.id} and ${modelId}`);

    try {
      const response = await client.mutate({
        mutation: DELETE_FROM_GRAPH,
        variables: { 
          model_id: modelId, 
          layer_id: selectedNode.id
        },
        fetchPolicy: 'no-cache'
      });

      console.log('Delete response:', response.data);

      if (!response.data?.deleteInModuleGraph) {
        throw new Error('Failed to delete layer - no data returned');
      }

      const graphData = response.data.deleteInModuleGraph;
      updateNodesFromGraphData(graphData);
      
      // Dispatch event for external listeners
      dispatch('layerDeleted', { layerId: selectedNode.id });

      selectedNode = null;
      saveStateToStorage();

    } catch (err) {
      console.error('Error deleting layer:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Unknown error occurred while deleting layer';
      }
    } finally {
      loading = false;
    }
  }
  
  async function deleteSelectedEdge() {
    if (!selectedEdge || !modelId) return;
    
    loading = true;
    error = null;

    console.log(`Disconnecting layers: ${selectedEdge.from} -> ${selectedEdge.to}`);

    try {
      const response = await client.mutate({
        mutation: DISCONNECT_NODES,
        variables: { 
          source_layer_id: selectedEdge.from, 
          target_layer_id: selectedEdge.to 
        },
        fetchPolicy: 'no-cache'
      });

      console.log('Disconnect response:', response.data);

      if (!response.data?.disconnectInModuleGraph) {
        throw new Error('Failed to disconnect layers - no data returned');
      }

      const graphData = response.data.disconnectInModuleGraph;
      updateNodesFromGraphData(graphData);
      
      // Dispatch event for external listeners
      dispatch('connectionDeleted', { connectionId: selectedEdge.id });

      selectedEdge = null;
      saveStateToStorage();

    } catch (err) {
      console.error('Error disconnecting layers:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Unknown error occurred while disconnecting layers';
      }
    } finally {
      loading = false;
    }
  }
  
  async function buildModuleGraph() {
    if (!modelId) {
      error = 'Model ID is missing';
      return;
    }
    
    loading = true;
    error = null;

    console.log('Building module graph...');

    try {
      const response = await client.mutate({
        mutation: BUILD_MODULE_GRAPH,
        variables: {},
        fetchPolicy: 'no-cache'
      });

      console.log('Build graph response:', response.data);

      if (!response.data?.buildModuleGraph) {
        throw new Error('Failed to build module graph - no data returned');
      }

      const graphData = response.data.buildModuleGraph;

      buildResult = graphData;
      
      // Update the local state with the built and sorted graph
      if (graphData.module_graph) {
        updateNodesFromGraphData(graphData.module_graph);
      }

      saveStateToStorage();

    } catch (err) {
      console.error('Error building module graph:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Unknown error occurred while building module graph';
      }
    } finally {
      loading = false;
    }
  }
  
  function clearSelection() {
    selectedLayerType = null;
    selectedNode = null;
    selectedEdge = null;
    error = null;
  }
  
  // Enhanced drag functionality
  function startDrag(node: Node, event: MouseEvent) {
    event.preventDefault();
    event.stopPropagation();
    
    // Determine drag mode based on modifier keys
    if (event.shiftKey || event.ctrlKey) {
      // Start connection drag
      dragMode = 'connect';
      dragStartNode = node;
      showConnectionPreview = true;
      
      const rect = canvas!.getBoundingClientRect();
      const nodeWidth = 100;
      const nodeHeight = 80;
      
      connectionPreviewStart = {
        x: node.x + nodeWidth,
        y: node.y + nodeHeight / 2
      };
      connectionPreviewEnd = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
    } else {
      // Start move drag
      dragMode = 'move';
      selectedNode = node;
      
      const rect = canvas!.getBoundingClientRect();
      dragOffset.x = event.clientX - rect.left - node.x;
      dragOffset.y = event.clientY - rect.top - node.y;
    }
    
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('mouseup', stopDrag);
  }

  function handleDrag(event: MouseEvent) {
    if (dragMode === 'none') return;
    
    const rect = canvas!.getBoundingClientRect();
    
    if (dragMode === 'move' && selectedNode) {
      // Move node
      selectedNode.x = event.clientX - rect.left - dragOffset.x;
      selectedNode.y = event.clientY - rect.top - dragOffset.y;
      
      // Constrain to canvas bounds
      const nodeWidth = 100;
      const nodeHeight = 80;
      selectedNode.x = Math.max(0, Math.min(rect.width - nodeWidth, selectedNode.x));
      selectedNode.y = Math.max(0, Math.min(rect.height - nodeHeight, selectedNode.y));
      
      // Trigger reactivity
      nodes = [...nodes];
      edges = [...edges];
    } else if (dragMode === 'connect') {
      // Update connection preview
      connectionPreviewEnd = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
    }
  }

  function stopDrag(event?: MouseEvent) {
    if (dragMode === 'connect' && event && dragStartNode) {
      const targetNode = getNodeAtPosition(event);
      if (targetNode && targetNode.id !== dragStartNode.id) {
        createConnection(dragStartNode.id, targetNode.id);
      }
    }

    // Only save state if we actually moved something
    if (dragMode === 'move') {
      saveStateToStorage();
    }

    // Reset drag state
    dragMode = 'none';
    dragStartNode = null;
    showConnectionPreview = false;

    document.removeEventListener('mousemove', handleDrag);
    document.removeEventListener('mouseup', stopDrag);

    // Force reactivity on nodes array
    nodes = [...nodes];
  }

  function getNodeAtPosition(event: MouseEvent): Node | null {
    const rect = canvas!.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    return nodes.find(node => {
      const nodeWidth = 100;
      const nodeHeight = 80;
      return x >= node.x && x <= node.x + nodeWidth &&
             y >= node.y && y <= node.y + nodeHeight;
    }) || null;
  }

  function getEdgePath(edge: Edge) {
    const fromNode = nodes.find(n => n.id === edge.from);
    const toNode = nodes.find(n => n.id === edge.to);
    
    if (!fromNode || !toNode) return '';
    
    // Calculate connection points on the edges of nodes
    const nodeWidth = 100;
    const nodeHeight = 80;
    
    const fromX = fromNode.x + nodeWidth; // Right edge of source node
    const fromY = fromNode.y + nodeHeight / 2; // Middle of source node
    const toX = toNode.x; // Left edge of target node
    const toY = toNode.y + nodeHeight / 2; // Middle of target node
    
    // Create a smooth curve
    const dx = toX - fromX;
    const controlOffset = Math.abs(dx) * 0.5; // Adjust curve intensity
    
    const controlX1 = fromX + controlOffset;
    const controlY1 = fromY;
    const controlX2 = toX - controlOffset;
    const controlY2 = toY;
    
    return `M ${fromX} ${fromY} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${toX} ${toY}`;
  }

  function getConnectionPreviewPath() {
    const controlOffset = Math.abs(connectionPreviewEnd.x - connectionPreviewStart.x) * 0.5;
    const controlX1 = connectionPreviewStart.x + controlOffset;
    const controlY1 = connectionPreviewStart.y;
    const controlX2 = connectionPreviewEnd.x - controlOffset;
    const controlY2 = connectionPreviewEnd.y;
    
    return `M ${connectionPreviewStart.x} ${connectionPreviewStart.y} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${connectionPreviewEnd.x} ${connectionPreviewEnd.y}`;
  }

  // Save state to sessionStorage
  function saveStateToStorage() {
    if (typeof window === 'undefined') return;
    
    const state = {
      nodes,
      edges,
      buildResult,
      nodeCounter
    };
    
    sessionStorage.setItem(`graph-state-${modelId}`, JSON.stringify(state));
  }

  // Load state from sessionStorage
  function loadStateFromStorage() {
    if (typeof window === 'undefined' || !modelId) return;
    
    const saved = sessionStorage.getItem(`graph-state-${modelId}`);
    if (saved) {
      try {
        const state = JSON.parse(saved);
        nodes = state.nodes || [];
        edges = state.edges || [];
        buildResult = state.buildResult || null;
        nodeCounter = state.nodeCounter || 1;
      } catch (err) {
        console.error('Error loading saved state:', err);
      }
    }
  }

  onMount(() => {
    if (modelId) {
      loadStateFromStorage();
    }
  });
</script>

<div class="graph-editor">
  <!-- Left Sidebar -->
  <div class="sidebar left">
    <h3>Layer Types</h3>
    {#each layerTypes as layerType}
      <button 
        class="block-button {selectedLayerType?.type === layerType.type ? 'selected' : ''}"
        style="border-left: 4px solid {layerType.color}"
        on:click={() => selectLayerType(layerType)}
      >
        {layerType.label}
      </button>
    {/each}
  </div>
  
  <!-- Canvas -->
  <div class="canvas" bind:this={canvas} on:click={clearSelection}>
    <!-- Build Graph Button -->
    <div class="build-graph-controls">
      <button 
        class="build-graph-button" 
        on:click={buildModuleGraph}
        disabled={loading}
      >
        {loading ? 'Building...' : 'Build & Sort Graph'}
      </button>
      {#if error}
        <div class="error-message">{error}</div>
      {/if}
    </div>
    <svg width="100%" height="100%">
      <!-- Edges -->
      {#each edges as edge}
        <path
          d={getEdgePath(edge)}
          stroke="#374151"
          stroke-width="2"
          fill="none"
          class="edge {selectedEdge?.id === edge.id ? 'selected' : ''}"
          on:click={(e) => selectEdge(edge, e)}
        />
      {/each}
    </svg>
    
    <!-- Nodes -->
    {#each nodes as node}
      <div
        class="node {selectedNode?.id === node.id || node.selected ? 'selected' : ''}"
        style="left: {node.x}px; top: {node.y}px; border-color: {node.color}"
        on:click={(e) => selectNode(node, e)}
        on:mousedown={(e) => startDrag(node, e)}
      >
        <div class="node-header" style="background-color: {node.color}">
          {node.layerConfig.linear?.name || node.layerConfig.conv2d?.name || node.layerConfig.conv1d?.name || node.id}
        </div>
        <div class="node-body">
          <div class="node-type">{node.layerConfig.type}</div>
        </div>
      </div>
    {/each}
  </div>
  
  <!-- Right Sidebar -->
  <div class="sidebar right">
    {#if selectedLayerType}
      <h3>Add {selectedLayerType.label}</h3>
      <div class="form">
        <label>
          Layer Name:
          <input type="text" bind:value={layerName} placeholder="Optional custom name">
        </label>
        
        {#if selectedLayerType.type === 'linear'}
          <label>
            Input Features:
            <input type="text" bind:value={inFeatures} placeholder="Enter input features">
          </label>
          <label>
            Output Features:
            <input type="text" bind:value={outFeatures} placeholder="Enter output features">
          </label>
          <label>
            <input type="checkbox" bind:checked={bias}>
            Bias
          </label>
        {:else if selectedLayerType.type === 'conv2d' || selectedLayerType.type === 'conv1d'}
          <label>
            Input Channels:
            <input type="text" bind:value={inChannels} placeholder="Enter input channels">
          </label>
          <label>
            Output Channels:
            <input type="text" bind:value={outChannels} placeholder="Enter output channels">
          </label>
          <label>
            Kernel Size:
            <input type="text" bind:value={kernelSize} placeholder={selectedLayerType.type === 'conv2d' ? "e.g., 3,3" : "e.g., 3"}>
          </label>
          <label>
            Stride (optional):
            <input type="text" bind:value={stride} placeholder={selectedLayerType.type === 'conv2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            Padding (optional):
            <input type="text" bind:value={padding} placeholder={selectedLayerType.type === 'conv2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            Dilation (optional):
            <input type="text" bind:value={dilation} placeholder={selectedLayerType.type === 'conv2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            Groups:
            <input type="text" bind:value={groups} placeholder="Default: 1">
          </label>
          <label>
            Padding Mode:
            <select bind:value={paddingMode}>
              <option value="zeros">zeros</option>
              <option value="reflect">reflect</option>
              <option value="replicate">replicate</option>
              <option value="circular">circular</option>
            </select>
          </label>
          <label>
            <input type="checkbox" bind:checked={convBias}>
            Bias
          </label>
        {/if}
        
        <button class="add-button" on:click={addNodeToGraph}>Add to Graph</button>
      </div>
    {:else if selectedNode}
      <h3>Selected Layer Configuration</h3>
      <div class="node-details">
        <p><strong>Type:</strong> {selectedNode.layerConfig.type}</p>
        <div class="form">
          <label>
            Layer Name:
            <input type="text" bind:value={layerName}>
          </label>
          
          {#if selectedNode.layerConfig.type === 'linear'}
            <label>
              Input Features:
              <input type="text" bind:value={inFeatures}>
            </label>
            <label>
              Output Features:
              <input type="text" bind:value={outFeatures}>
            </label>
            <label>
              <input type="checkbox" bind:checked={bias}>
              Bias
            </label>
          {:else if selectedNode.layerConfig.type === 'conv2d' || selectedNode.layerConfig.type === 'conv1d'}
            <label>
              Input Channels:
              <input type="text" bind:value={inChannels}>
            </label>
            <label>
              Output Channels:
              <input type="text" bind:value={outChannels}>
            </label>
            <label>
              Kernel Size:
              <input type="text" bind:value={kernelSize}>
            </label>
            <label>
              Stride:
              <input type="text" bind:value={stride}>
            </label>
            <label>
              Padding:
              <input type="text" bind:value={padding}>
            </label>
            <label>
              Dilation:
              <input type="text" bind:value={dilation}>
            </label>
            <label>
              Groups:
              <input type="text" bind:value={groups}>
            </label>
            <label>
              Padding Mode:
              <select bind:value={paddingMode}>
                <option value="zeros">zeros</option>
                <option value="reflect">reflect</option>
                <option value="replicate">replicate</option>
                <option value="circular">circular</option>
              </select>
            </label>
            <label>
              <input type="checkbox" bind:checked={convBias}>
              Bias
            </label>
          {/if}
          
          <div class="button-group">
            <button class="delete-button" on:click={deleteSelectedNode}>Delete</button>
          </div>
        </div>
      </div>
    {:else if selectedEdge}
      <h3>Connection</h3>
      <div class="edge-details">
        {#each [selectedEdge.from, selectedEdge.to] as nodeId, index}
          {@const node = nodes.find(n => n.id === nodeId)}
          {#if node}
            <p><strong>{index === 0 ? 'From' : 'To'}:</strong> 
              {node.layerConfig.linear?.name || node.layerConfig.conv2d?.name || node.layerConfig.conv1d?.name || nodeId} 
              <span class="node-type">({node.layerConfig.type})</span>
            </p>
          {:else}
            <p><strong>{index === 0 ? 'From' : 'To'}:</strong> {nodeId} <span class="node-type">(unknown)</span></p>
          {/if}
        {/each}
        <button class="delete-button" on:click={deleteSelectedEdge}>Delete Connection</button>
      </div>
    {:else}
      <h3>Instructions</h3>
      <div class="instructions">
        <p>• Select a layer type to add new layers</p>
        <p>• Click on layers to select and edit them</p>
        <p>• Click on connections to select and delete them</p>
        <p>• <strong>Shift + Drag</strong> to add connection between layers</p>
        <p>• <strong>Drag</strong> layers to reposition them</p>
        
        {#if buildResult}
          <div class="build-result">
            <h4>Built Graph Successfully</h4>
            <p><strong>Layers:</strong> {buildResult.module_graph.layers.length}</p>
            <p><strong>Connections:</strong> {buildResult.module_graph.edges.length}</p>
            
            {#if buildResult.module_graph.sorted && buildResult.module_graph.layers.length > 0}
              <div class="layer-order">
                <strong>Layer Execution Order:</strong>
                <ol>
                  {#each buildResult.module_graph.layers as layer}
                    <li>{layer.name} ({layer.type})</li>
                  {/each}
                </ol>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>