<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { page } from '$app/stores';
  import { onMount, tick} from 'svelte';
  import { SvelteFlow } from '@xyflow/svelte';
  import client from '$lib/apolloClient';
  import { ADD_TO_GRAPH, CONNECT_NODES, DELETE_FROM_GRAPH, DISCONNECT_NODES,
          BUILD_MODULE_GRAPH } from '$lib/mutations';
  import {VALIDATE_GRAPH} from '$lib/mutations';
  import {GET_MODEL} from '$lib/queries';
  import type {Model} from '../../../../../../source/types/modelTypes';
  import type { LayerConfigInput } from '../../../../../../source/types/layerTypes';
  import LayerForm from './LayerForm.svelte';
  import '@xyflow/svelte/dist/style.css';
  import './graph-construction.css';

  const dispatch = createEventDispatcher<{
    layerAdded: { layerConfig: LayerConfigInput };
    layerDeleted: { layerId: string };
    connectionCreated: { fromId: string; toId: string };
    connectionDeleted: { connectionId: string };
  }>();

  let validationTrigger = 0;

  // Available layer types
  const layerTypes = [
    { type: 'linear', color: '#DC2626', label: 'Linear' },
    { type: 'conv2d', color: '#4F46E5', label: 'Conv2D' },
    { type: 'conv1d', color: '#7C3AED', label: 'Conv1D' },
    { type: 'convtranspose2d', color: '#E11D48', label: 'ConvTranspose2D' },
    { type: 'maxpool2d', color: '#0D9488', label: 'MaxPool2D' },
    { type: 'maxpool1d', color: '#14B8A6', label: 'MaxPool1D' },
    { type: 'avgpool2d', color: '#0EA5E9', label: 'AvgPool2D' },
    { type: 'avgpool1d', color: '#38BDF8', label: 'AvgPool1D' },
    { type: 'batchnorm2d', color: '#F59E0B', label: 'BatchNorm2D' },
    { type: 'batchnorm1d', color: '#FBBF24', label: 'BatchNorm1D' },
    { type: 'flatten', color: '#6B7280', label: 'Flatten' },
    { type: 'dropout', color: '#9CA3AF', label: 'Dropout' },
    { type: 'dropout2d', color: '#D1D5DB', label: 'Dropout2D' },
    { type: 'elu', color: '#9333EA', label: 'ELU' },
    { type: 'relu', color: '#10B981', label: 'ReLU' },
    { type: 'leakyrelu', color: '#8B5CF6', label: 'LeakyReLU' },
    { type: 'sigmoid', color: '#EF4444', label: 'Sigmoid' },
    { type: 'logsigmoid', color: '#F87171', label: 'LogSigmoid' },
    { type: 'tanh', color: '#6366F1', label: 'Tanh' },
    { type: 'cat', color: '#FB923C', label: 'Cat' }
  ];
  
  // SvelteFlow node and edge types
  interface FlowNode {
    id: string;
    type?: string;
    position: { x: number; y: number };
    data: {
      label: string;
      layerType: string;
      layerConfig: LayerConfigInput;
      color: string;
    };
    selected?: boolean;
  }

  interface FlowEdge {
    id: string;
    source: string;
    target: string;
    type?: string;
  }

  // State
  let nodes = $state.raw<FlowNode[]>([]);
  let edges = $state.raw<FlowEdge[]>([]);
  let selectedNode: FlowNode | null = $state(null);
  let selectedLayerType: typeof layerTypes[number] | null = $state(null);
  let selectedEdge: FlowEdge | null = $state(null);
  let nodeCounter = 1;
  let modelId: string | null = null;
  let loading = $state(false);
  let error = $state<string | null>(null);
  let buildResult = $state<any>(null);
  let modelDetails: Model | null = null;
  let graphValidationResult = $state<any>(null);
  
  // New state for details view
  let showNodeDetails = $state(false);
  let detailsNode: FlowNode | null = $state(null);
  let lastClickTime = 0;
  const DOUBLE_CLICK_DELAY = 300; // ms

  // Extract modelId from URL
  $effect(() => {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    modelId = (modelIndex !== -1 && modelIndex + 1 < pathParts.length) ? pathParts[modelIndex + 1] : null;
  });

  fetchModelDetails();

  async function fetchModelDetails() {
    try {
      const response = await client.query({
        query: GET_MODEL,
        fetchPolicy: 'network-only'
      });
      
      modelDetails = response.data?.getModel;
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }

  $effect(() => {
    if (modelId && typeof window !== 'undefined') {
      loadStateFromStorage();
    }
  });
  
  function selectLayerType(layerType: typeof layerTypes[number]) {
    selectedLayerType = layerType;
    selectedNode = null;
    selectedEdge = null;
  }

  async function handleAddLayer(event: CustomEvent<{ layerConfig: LayerConfigInput }>) {
    const { layerConfig } = event.detail;
    
    loading = true;
    error = null;
    
    try {
      const response = await client.mutate({
        mutation: ADD_TO_GRAPH,
        variables: { layer_config: layerConfig },
        fetchPolicy: 'no-cache'
      });

      console.log('Response from add layer:', response);

      if (!response.data?.appendToModuleGraph) {
        throw new Error('Failed to add layer to graph - no data returned');
      }

      const graphData = response.data.appendToModuleGraph;
      updateNodesFromGraphData(graphData);
      
      nodeCounter++;
      saveStateToStorage();
      dispatch('layerAdded', { layerConfig });

    } catch (err) {
      console.error('Error adding layer to graph:', err);
      error = err instanceof Error ? err.message : 'Unknown error occurred while adding layer';
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
          selectable: true,
          position: {
            x: existingNode?.position.x || 300,
            y: existingNode?.position.y || 200 + (index * 120)
          },
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
      edges = [];
      graphData.edges.forEach((edge: any) => {
        if (edge.target_ids && Array.isArray(edge.target_ids)) {
          edge.target_ids.forEach((targetId: string) => {
            edges.push({
              id: `${edge.source_id}-${targetId}`,
              source: edge.source_id,
              target: targetId
            });
          });
        }
      });
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
    } else if (layer.type === 'convtranspose2d') {
      return {
        type: 'convtranspose2d',
        convtranspose2d: {
          name: layer.name,
          in_channels: layer.in_channels,
          out_channels: layer.out_channels,
          kernel_size: layer.kernel_size,
          stride: layer.stride || 1,
          padding: layer.padding || 0,
          output_padding: layer.output_padding || [0, 0],
          groups: layer.groups || 1,
          bias: layer.bias ?? true
        }
      }
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
    } else if (layer.type === 'maxpool2d') {
      return {
        type: 'maxpool2d',
        maxpool2d: {
          name: layer.name,
          kernel_size: layer.kernel_size,
          stride: layer.stride,
          padding: layer.padding,
          dilation: layer.dilation,
          return_indices: layer.return_indices ?? false,
          ceil_mode: layer.ceil_mode ?? false
        }
      };
    } else if (layer.type === 'avgpool1d') {
      return {
        type: 'avgpool1d',
        avgpool1d: {
          name: layer.name,
          kernel_size: layer.kernel_size,
          stride: layer.stride,
          padding: layer.padding,
          count_include_pad: layer.count_include_pad ?? false,
          divisor_override: layer.divisor_override || ''
        }
      };
    } else if (layer.type === 'avgpool2d') {
      return {
        type: 'avgpool2d',
        avgpool2d: {
          name: layer.name,
          kernel_size: layer.kernel_size,
          stride: layer.stride,
          padding: layer.padding,
          count_include_pad: layer.count_include_pad ?? false,
          divisor_override: layer.divisor_override || ''
        }
      };
    } else if (layer.type === 'batchnorm1d') {
      return {
        type: 'batchnorm1d',
        batchnorm1d: {
          name: layer.name,
          num_features: layer.num_features,
          eps: layer.eps || 1e-5,
          momentum: layer.momentum || 0.1,
          affine: layer.affine ?? true,
          track_running_status: layer.track_running_stats ?? true
        }
      };
    } else if (layer.type === 'batchnorm2d') {
      return {
        type: 'batchnorm2d',
        batchnorm2d: {
          name: layer.name,
          num_features: layer.num_features,
          eps: layer.eps || 1e-5,
          momentum: layer.momentum || 0.1,
          affine: layer.affine ?? true,
          track_running_status: layer.track_running_stats ?? true
        }
      };
    } else if(layer.type === 'flatten') {
      return {
        type: 'flatten',
        flatten: {
          name: layer.name,
          start_dim: layer.start_dim || 1,
          end_dim: layer.end_dim || -1
        }
      };
    } else if (layer.type === 'dropout') {
      return {
        type: 'dropout',
        dropout: {
          name: layer.name,
          p: layer.p || 0.5
        }
      };
    } else if (layer.type === 'dropout2d') {
      return {
        type: 'dropout2d',
        dropout2d: {
          name: layer.name,
          p: layer.p || 0.5
        }
      };
    } else if (layer.type === 'elu') {
      return {
        type: 'elu',
        elu: {
          name: layer.name,
          alpha: layer.alpha || 1.0,
          inplace: layer.inplace ?? false
        }
      };
    } else if (layer.type === 'relu') {
      return {
        type: 'relu',
        relu: {
          name: layer.name,
          inplace: layer.inplace ?? false
        }
      };
    } else if (layer.type === 'leakyrelu') {
      return {
        type: 'leakyrelu',
        leakyrelu: {
          name: layer.name,
          negative_slope: layer.negative_slope || 0.01,
          inplace: layer.inplace ?? false
        }
      };
    } else if (layer.type === 'sigmoid') {
      return { type: 'sigmoid',
        sigmoid: {
          name: layer.name,
        } };
    } else if (layer.type === 'logsigmoid') {
      return { type: 'logsigmoid',
        logsigmoid: {
          name: layer.name,
        } };
    } else if (layer.type === 'tanh') {
      return { type: 'tanh',
        tanh: {
          name: layer.name,
        } };
    } else if (layer.type === 'cat') {
      return {
        type: 'cat',
        cat: {
          name: layer.name,
          dimension: layer.dimension || 0
        }
      };
    }

    throw new Error(`Unsupported layer type: ${layer.type}`);
  }

  
  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Escape' && showNodeDetails) {
      closeNodeDetails();
      return;
    }
    
    if ((event.key === 'Delete' || event.key === 'Backspace') && !loading) {
      console.log('Delete key pressed');
      console.log('Current selectedEdge:', selectedEdge);
      console.log('Current selectedNode:', selectedNode);
      if (selectedNode) {
        handleDeleteNode();
      } else if (selectedEdge) {
        console.log('Selected edge:', selectedEdge);
        handleDeleteEdge();
      } else {
        console.log('No node or edge selected');
      }
    }
  }

  function onNodeClick(event: any) {
    console.log('Node clicked:', event.detail?.node || event.node);
    const node = event.detail?.node || event.node;
    
    if (node) {
      const currentTime = Date.now();
      
      // Check for double-click
      if (currentTime - lastClickTime < DOUBLE_CLICK_DELAY && selectedNode?.id === node.id) {
        // Double-click detected
        showNodeDetailsView(node);
      } else {
        // Single click
        selectedNode = node;
        selectedLayerType = null;
        selectedEdge = null;
      }
      
      lastClickTime = currentTime;
      console.log('Selected node set to:', selectedNode);
    }
  }

  function showNodeDetailsView(node: FlowNode) {
    detailsNode = node;
    showNodeDetails = true;
  }

  function closeNodeDetails() {
    showNodeDetails = false;
    detailsNode = null;
  }

  function getLayerDetails(layerConfig: LayerConfigInput) {
    const getName = (name?: string) => name || 'Unnamed';

    if (layerConfig.type === 'linear' && layerConfig.linear) {
      return {
        'Layer Type': 'Linear',
        'Name': getName(layerConfig.linear.name),
        'Input Features': layerConfig.linear.in_features,
        'Output Features': layerConfig.linear.out_features
      };
    }

    if (layerConfig.type === 'conv2d' && layerConfig.conv2d) {
      return {
        'Layer Type': 'Conv2D',
        'Name': getName(layerConfig.conv2d.name),
        'Input Channels': layerConfig.conv2d.in_channels,
        'Output Channels': layerConfig.conv2d.out_channels,
        'Kernel Size': Array.isArray(layerConfig.conv2d.kernel_size)
          ? layerConfig.conv2d.kernel_size.join(' x ')
          : layerConfig.conv2d.kernel_size
      };
    }

    if (layerConfig.type === 'conv1d' && layerConfig.conv1d) {
      return {
        'Layer Type': 'Conv1D',
        'Name': getName(layerConfig.conv1d.name),
        'Input Channels': layerConfig.conv1d.in_channels,
        'Output Channels': layerConfig.conv1d.out_channels,
        'Kernel Size': layerConfig.conv1d.kernel_size
      };
    }

    if (layerConfig.type === 'convtranspose2d' && layerConfig.convtranspose2d) {
      return {
        'Layer Type': 'ConvTranspose2D',
        'Name': getName(layerConfig.convtranspose2d.name),
        'Input Channels': layerConfig.convtranspose2d.in_channels,
        'Output Channels': layerConfig.convtranspose2d.out_channels,
        'Kernel Size': Array.isArray(layerConfig.convtranspose2d.kernel_size)
          ? layerConfig.convtranspose2d.kernel_size.join(' x ')
          : layerConfig.convtranspose2d.kernel_size
      };
    }

    if (layerConfig.type === 'maxpool2d' && layerConfig.maxpool2d) {
      return {
        'Layer Type': 'MaxPool2D',
        'Name': getName(layerConfig.maxpool2d.name),
        'Kernel Size': layerConfig.maxpool2d.kernel_size
      };
    }

    if (layerConfig.type === 'maxpool1d' && layerConfig.maxpool1d) {
      return {
        'Layer Type': 'MaxPool1D',
        'Name': getName(layerConfig.maxpool1d.name),
        'Kernel Size': layerConfig.maxpool1d.kernel_size
      };
    }

    if (layerConfig.type === 'avgpool2d' && layerConfig.avgpool2d) {
      return {
        'Layer Type': 'AvgPool2D',
        'Name': getName(layerConfig.avgpool2d.name),
        'Kernel Size': layerConfig.avgpool2d.kernel_size
      };
    }

    if (layerConfig.type === 'avgpool1d' && layerConfig.avgpool1d) {
      return {
        'Layer Type': 'AvgPool1D',
        'Name': getName(layerConfig.avgpool1d.name),
        'Kernel Size': layerConfig.avgpool1d.kernel_size
      };
    }

    if (layerConfig.type === 'batchnorm2d' && layerConfig.batchnorm2d) {
      return {
        'Layer Type': 'BatchNorm2D',
        'Name': getName(layerConfig.batchnorm2d.name),
        'Num Features': layerConfig.batchnorm2d.num_features
      };
    }

    if (layerConfig.type === 'batchnorm1d' && layerConfig.batchnorm1d) {
      return {
        'Layer Type': 'BatchNorm1D',
        'Name': getName(layerConfig.batchnorm1d.name),
        'Num Features': layerConfig.batchnorm1d.num_features
      };
    }

    if (layerConfig.type === 'flatten' && layerConfig.flatten) {
      return {
        'Layer Type': 'Flatten',
        'Name': getName(layerConfig.flatten.name)
      };
    }

    if (layerConfig.type === 'dropout' && layerConfig.dropout) {
      return {
        'Layer Type': 'Dropout',
        'Name': getName(layerConfig.dropout.name)
      };
    }

    if (layerConfig.type === 'dropout2d' && layerConfig.dropout2d) {
      return {
        'Layer Type': 'Dropout2D',
        'Name': getName(layerConfig.dropout2d.name)
      };
    }

    if (layerConfig.type === 'elu' && layerConfig.elu) {
      return {
        'Layer Type': 'ELU',
        'Name': getName(layerConfig.elu.name)
      };
    }

    if (layerConfig.type === 'relu' && layerConfig.relu) {
      return {
        'Layer Type': 'ReLU',
        'Name': getName(layerConfig.relu.name)
      };
    }

    if (layerConfig.type === 'leakyrelu' && layerConfig.leakyrelu) {
      return {
        'Layer Type': 'LeakyReLU',
        'Name': getName(layerConfig.leakyrelu.name)
      };
    }

    if (layerConfig.type === 'sigmoid' && layerConfig.sigmoid) {
      return {
        'Layer Type': 'Sigmoid',
        'Name': getName(layerConfig.sigmoid.name)
      };
    }

    if (layerConfig.type === 'logsigmoid' && layerConfig.logsigmoid) {
      return {
        'Layer Type': 'LogSigmoid',
        'Name': getName(layerConfig.logsigmoid.name)
      };
    }

    if (layerConfig.type === 'tanh' && layerConfig.tanh) {
      return {
        'Layer Type': 'Tanh',
        'Name': getName(layerConfig.tanh.name)
      };
    }

    if (layerConfig.type === 'cat' && layerConfig.cat) {
      return {
        'Layer Type': 'Cat',
        'Name': getName(layerConfig.cat.name)
      };
    }

    return { 'Layer Type': 'Unknown' };
  }

  function onEdgeClick(event: any) {
    console.log('Edge clicked - full event:', event);
    event.stopPropagation?.(); // Prevent event bubbling
    
    const edge = event.detail?.edge || event.edge;
    console.log('Extracted edge:', edge);
    
    if (edge) {
      selectedEdge = {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type
      };
      selectedNode = null;
      selectedLayerType = null;
      console.log('Selected edge set to:', selectedEdge);
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
    // Handle node position changes
    const changes = event.detail;
    changes.forEach(change => {
      if (change.type === 'position' && change.position) {
        const nodeIndex = nodes.findIndex(n => n.id === change.id);
        if (nodeIndex !== -1) {
          nodes[nodeIndex].position = change.position;
        }
      }
    });
    saveStateToStorage();
  }

  async function createConnection(fromId: string, toId: string) {

    loading = true;
    error = null;

    try {
      const response = await client.mutate({
        mutation: CONNECT_NODES,
        variables: { 
          source_layer_id: fromId, 
          target_layer_id: toId 
        },
        fetchPolicy: 'no-cache'
      });

      console.log('Response from connect nodes:', response);

      if (!response.data?.connectInModuleGraph) {
        throw new Error('Failed to connect layers - no data returned');
      }

      const graphData = response.data.connectInModuleGraph;
      updateNodesFromGraphData(graphData);
      
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
  
  async function handleDeleteNode() {
    if (!selectedNode){
      error = 'No layer selected for deletion';
      return;
    } 
    loading = true;
    error = null;

    try {
      const response = await client.mutate({
        mutation: DELETE_FROM_GRAPH,
        variables: {
          layer_id: selectedNode.id
        },
        fetchPolicy: 'no-cache'
      });

      console.log('Response from delete layer:', response);

      if (!response.data?.deleteInModuleGraph) {
        throw new Error('Failed to delete layer - no data returned');
      }

      const graphData = response.data.deleteInModuleGraph;
      updateNodesFromGraphData(graphData);
      
      dispatch('layerDeleted', { layerId: selectedNode.id });
      selectedNode = null;
      saveStateToStorage();

    } catch (err) {
      console.error('Error deleting layer:', err);
      error = err instanceof Error ? err.message : 'Unknown error occurred while deleting layer';
    } finally {
      loading = false;
    }
  }
  
  async function handleDeleteEdge() {
    if (!selectedEdge ){
      console.warn('No edge selected for deletion');
      return;
    }
    
    loading = true;
    error = null;

    try {
      const response = await client.mutate({
        mutation: DISCONNECT_NODES,
        variables: { 
          source_layer_id: selectedEdge.source, 
          target_layer_id: selectedEdge.target 
        },
        fetchPolicy: 'no-cache'
      });

      console.log('Response from disconnect nodes:', response);

      if (!response.data?.disconnectInModuleGraph) {
        throw new Error('Failed to disconnect layers - no data returned');
      }

      const graphData = response.data.disconnectInModuleGraph;
      updateNodesFromGraphData(graphData);
      
      dispatch('connectionDeleted', { connectionId: selectedEdge.id });
      selectedEdge = null;
      saveStateToStorage();

    } catch (err) {
      console.error('Error disconnecting layers:', err);
      error = err instanceof Error ? err.message : 'Unknown error occurred while disconnecting layers';
    } finally {
      loading = false;
    }
  }
  
  async function buildModuleGraph() {
    loading = true;
    error = null;

    try {
      const response = await client.mutate({
        mutation: BUILD_MODULE_GRAPH,
        variables: {},
        fetchPolicy: 'no-cache'
      });

      if (!response.data?.buildModuleGraph) {
        throw new Error('Failed to build module graph - no data returned');
      }

      const graphData = response.data.buildModuleGraph;
      buildResult = graphData;

      await validateGraphStructure();
      
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

  async function validateGraphStructure() {
    try {
      let inputDimension: number[];
      const datasetType = modelDetails?.dataset_config?.name?.toLowerCase();
      switch (datasetType) {
        case 'mnist':
          inputDimension = [1, 28, 28];
          break;
        case 'cifar10':
          inputDimension = [3, 32, 32];
          break;
        case 'image_folder':
          inputDimension = [3, 224, 224];
          break;
        default:
          inputDimension = [1, 28, 28];
      }

      const response = await client.mutate({
        mutation: VALIDATE_GRAPH,
        variables: { in_dimension: inputDimension },
        fetchPolicy: 'no-cache',
        errorPolicy: 'all'
      });

      console.log('Response from validate graph:', response);

      if (response.data?.validateModuleGraph) {
        graphValidationResult = { ...response.data.validateModuleGraph, status: response.data.validateModuleGraph.status || [] };
        validationTrigger++;
        await tick();
      } else {
        graphValidationResult = null;
        validationTrigger++;
      }
    } catch (err) {
      console.error('Validation error:', err);
      graphValidationResult = null;
      validationTrigger++;
    }
  }

  onMount(() => {
    if (modelId) {
      loadStateFromStorage();
    }
    document.addEventListener('keydown', handleKeyDown);
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
        onclick={() => selectLayerType(layerType)}
      >
        {layerType.label}
      </button>
    {/each}
  </div>
  
  <!-- SvelteFlow Canvas -->
  <div
    class="canvas-container"
    role="button"
    tabindex="0"
    aria-label="Clear selection"
    onclick={(e) => {
      if (e.target === e.currentTarget) {
        clearSelection();
      }
    }}
    onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { clearSelection(); } }}
  >
    <!-- Build Graph Button -->
    <div class="build-graph-controls">
      <button 
        class="build-graph-button" 
        onclick={buildModuleGraph}
        disabled={loading}
      >
        {loading ? 'Building...' : 'Build & Sort Graph'}
      </button>
      {#if error}
        <div class="error-message">{error}</div>
      {/if}
    </div>

    <div style="width: 100%; height: 100%;">
      <SvelteFlow 
        bind:nodes 
        bind:edges
        onnodeclick={onNodeClick}
        onedgeclick={onEdgeClick}
        onconnect={onConnect}
        onnodeschange={onNodesChange}
        fitView
        nodeTypes={{}}
        edgeTypes={{}}
      >
        <!-- Custom Node Template -->
        <template slot="node" let:node>
          <div
            class="custom-node {selectedNode?.id === node.id ? 'selected' : ''}"
            style="border: 2px solid {node.data.color} !important;"
          >
            <div class="node-header" style="background-color: {node.data.color}">
              {node.data.label}
            </div>
            <div class="node-body">
              <div class="node-type">{node.data.layerType}</div>
            </div>
          </div>
        </template>
      </SvelteFlow>
    </div>

    <!-- Node Details Modal/Overlay -->
    {#if showNodeDetails && detailsNode}
      <div
        class="node-details-overlay"
        role="button"
        tabindex="0"
        aria-label="Close node details"
        onclick={closeNodeDetails}
        onkeydown={(e) => {
          if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') {
            closeNodeDetails();
          }
        }}
      >
        <div
          class="node-details-modal"
          role="dialog"
          aria-modal="true"
          tabindex="0"
          onclick={(e) => e.stopPropagation()}
          onkeydown={(e) => {
            if (e.key === 'Escape') {
              closeNodeDetails();
            }
          }}
        >
          <div class="details-header" style="background-color: {detailsNode.data.color}">
            <h3>{detailsNode.data.label}</h3>
            <button class="close-button" onclick={closeNodeDetails}>×</button>
          </div>
          
          <div class="details-content">
            {#each Object.entries(getLayerDetails(detailsNode.data.layerConfig)) as [key, value]}
              <div class="detail-row">
                <span class="detail-label">{key}:</span>
                <span class="detail-value">{value}</span>
              </div>
            {/each}
          </div>
          
          <div class="details-footer">
            <span class="hint">Press ESC to close</span>
          </div>
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Right Sidebar -->
  <LayerForm 
    {selectedLayerType}
    {nodes}
    {loading}
    {buildResult}
    {graphValidationResult}
    on:addLayer={handleAddLayer}
  />
</div>