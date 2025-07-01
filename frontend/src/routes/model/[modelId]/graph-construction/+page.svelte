<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { page } from '$app/stores';
  import { onMount, tick } from 'svelte';
  import client from '$lib/apolloClient';
  import { ADD_TO_GRAPH, CONNECT_NODES, DELETE_FROM_GRAPH, DISCONNECT_NODES,
          BUILD_MODULE_GRAPH } from '$lib/mutations';
  import {VALIDATE_GRAPH} from '$lib/mutations';
  import {GET_MODEL} from '$lib/queries';
  import type {Model} from '../../../../../../source/types/modelTypes';
  import type { LayerConfigInput, LinearLayerConfigInput, 
                Conv2dLayerConfigInput, Conv1dLayerConfigInput, 
                MaxPool2dLayerConfigInput, MaxPool1dLayerConfigInput,
                AvgPool2dLayerConfigInput, AvgPool1dLayerConfigInput, 
                BatchNorm2dLayerConfigInput, BatchNorm1dLayerConfigInput,
                FlattenLayerConfigInput, DropoutLayerConfigInput,
                ELULayerConfigInput, ReLULayerConfigInput,
                LeakyReLULayerConfigInput, SigmoidLayerConfigInput, 
                LogSigmoidLayerConfigInput, TanhLayerConfigInput} from '../../../../../../source/types/layerTypes';
  import './graph-construction.css';

  // Type-safe dispatch functions
  const dispatch = createEventDispatcher<{
    layerAdded: { layerConfig: LayerConfigInput };
    layerDeleted: { layerId: string };
    connectionCreated: { fromId: string; toId: string };
    connectionDeleted: { connectionId: string };
  }>();

  let validationResult: any = null;
  let validationTrigger = 0;

  // Available layer types
  const layerTypes = [
    { type: 'linear', color: '#DC2626', label: 'Linear' },
    { type: 'conv2d', color: '#4F46E5', label: 'Conv2D' },
    { type: 'conv1d', color: '#7C3AED', label: 'Conv1D' },
    { type: 'maxpool2d', color: '#0D9488', label: 'MaxPool2D' },
    { type: 'maxpool1d', color: '#14B8A6', label: 'MaxPool1D' },
    { type: 'avgpool2d', color: '#0EA5E9', label: 'AvgPool2D' },
    { type: 'avgpool1d', color: '#38BDF8', label: 'AvgPool1D' },
    { type: 'batchnorm2d', color: '#F59E0B', label: 'BatchNorm2D' },
    { type: 'batchnorm1d', color: '#FBBF24', label: 'BatchNorm1D' },
    { type: 'flatten', color: '#6B7280', label: 'Flatten' },
    { type: 'dropout', color: '#9CA3AF', label: 'Dropout' },
    { type: 'elu', color: '#9333EA', label: 'ELU' },
    { type: 'relu', color: '#10B981', label: 'ReLU' },
    { type: 'leakyrelu', color: '#8B5CF6', label: 'LeakyReLU' },
    { type: 'sigmoid', color: '#EF4444', label: 'Sigmoid' },
    { type: 'logsigmoid', color: '#F87171', label: 'LogSigmoid' },
    { type: 'tanh', color: '#6366F1', label: 'Tanh' }
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

  // MaxPool2d and MaxPool1d layer fields
  let poolKernelSize = '';
  let poolStride = '';
  let poolPadding = '';
  let poolDilation = '';
  let returnIndices = false;
  let ceilMode = false;

  // AvgPool2d and AvgPool1d layer fields
  let countIncludePad = false;
  let divisorOverride = '';

  // BatchNorm layer fields
  let numFeatures = '';
  let eps = '1e-05';
  let momentum = '0.1';
  let affine = true;
  let trackRunningStatus = false;

  // Flatten layer fields
  let startDim = '1';
  let endDim = '-1';

  // Dropout layer fields
  let dropoutP = '0.5';

  // ELU layer fields
  let alpha = '1.0';
  let eluInplace = false;

  // ReLU layer fields
  let reluInplace = false;

  // LeakyReLU layer fields
  let negativeSlope = '0.01';
  let leakyReluInplace = false;

  let modelDetails: Model | null = null;
  let graphValidationResult: any = null;

  // Extract modelId from URL
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    modelId = (modelIndex !== -1 && modelIndex + 1 < pathParts.length) ? pathParts[modelIndex + 1] : null;
  }

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
    inChannels = '';
    outChannels = '';
    kernelSize = '';
    stride = '';
    padding = '';
    dilation = '';
    groups = '1';
    poolKernelSize = '';
    poolStride = '';
    poolPadding = '';
    poolDilation = '';
    returnIndices = false;
    ceilMode = false;
    countIncludePad = false;
    divisorOverride = '';
    numFeatures = '';
    eps = '';
    momentum = '0.1';
    affine = false;
    trackRunningStatus = false;
    startDim = '1';
    endDim = '-1';
    dropoutP = '0.5';
    alpha = '1.0';
    eluInplace = false;
    reluInplace = false;
    negativeSlope = '0.01';
    leakyReluInplace = false;
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
      const kernelSizeArray = parseArrayInput(kernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "3")';
      }
    } else if (selectedLayerType.type === 'maxpool2d') {
      if (!poolKernelSize.trim()) {
        return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
        return 'Kernel size must be an array of exactly two numbers (e.g., "2,2")';
      }
    } else if (selectedLayerType.type === 'maxpool1d') {
      if (!poolKernelSize.trim()) {
        return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
        return 'Kernel size must be a number (e.g., "2")';
      }
    } else if (selectedLayerType.type === 'avgpool2d') {
      if (!poolKernelSize.trim()) {
          return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 2) {
          return 'Kernel size must be an array of exactly two numbers (e.g., "2,2")';
      }
    } else if (selectedLayerType.type === 'avgpool1d') {
      if (!poolKernelSize.trim()) {
          return 'Kernel size is required';
      }
      
      // Validate kernel size format
      const kernelSizeArray = parseArrayInput(poolKernelSize);
      if (!kernelSizeArray || kernelSizeArray.length !== 1) {
          return 'Kernel size must be a number (e.g., "2")';
      }
    } else if (selectedLayerType.type === 'batchnorm2d' || selectedLayerType.type === 'batchnorm1d') {
      const numFeaturesNum = Number(numFeatures);
      
      if (!numFeatures || isNaN(numFeaturesNum) || numFeaturesNum <= 0) {
          return 'Number of features must be a positive number';
      }
      
      const epsNum = Number(eps);
      if (isNaN(epsNum) || epsNum <= 0) {
          return 'Eps must be a positive number';
      }
      
      const momentumNum = Number(momentum);
      if (isNaN(momentumNum) || momentumNum < 0 || momentumNum > 1) {
          return 'Momentum must be between 0 and 1';
      }
    } else if (selectedLayerType.type === 'flatten') {
      const startDimNum = Number(startDim);
      const endDimNum = Number(endDim);
      
      if (isNaN(startDimNum)) {
        return 'Start dimension must be a valid number';
      }
      if (isNaN(endDimNum)) {
        return 'End dimension must be a valid number';
      }
    } else if (selectedLayerType.type === 'dropout') {
      const pNum = Number(dropoutP);
      
      if (isNaN(pNum) || pNum < 0 || pNum > 1) {
        return 'Dropout probability must be between 0 and 1';
      }
    } else if (selectedLayerType.type === 'elu') {
      const alphaNum = Number(alpha);
      
      if (isNaN(alphaNum) || alphaNum <= 0) {
        return 'Alpha must be a positive number';
      }
    } else if (selectedLayerType.type === 'leakyrelu') {
      const slopeNum = Number(negativeSlope);
      
      if (isNaN(slopeNum)) {
        return 'Negative slope must be a valid number';
      }
    } else if (selectedLayerType.type === 'sigmoid' || selectedLayerType.type === 'logsigmoid' || selectedLayerType.type === 'tanh') {
    // No additional validation needed for these layers
    }
    
    return null;
  }

  function createLayerConfig(): LayerConfigInput {
    console.log('Creating layer config for type:', selectedLayerType);
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
    } else if (selectedLayerType!.type === 'maxpool2d') {
      const maxpool2dConfig: MaxPool2dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2, 2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        dilation: poolDilation ? (parseArrayInput(poolDilation) ?? undefined) : undefined,
        return_indices: returnIndices,
        ceil_mode: ceilMode
      };
      
      return {
        type: 'maxpool2d',
        maxpool2d: maxpool2dConfig
      };
    } else if (selectedLayerType!.type === 'maxpool1d') {
      const maxpool1dConfig: MaxPool1dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        dilation: poolDilation ? (parseArrayInput(poolDilation) ?? undefined) : undefined,
        return_indices: returnIndices,
        ceil_mode: ceilMode
      };
      
      return {
        type: 'maxpool1d',
        maxpool1d: maxpool1dConfig
      };
    } else if (selectedLayerType!.type === 'avgpool2d') {
      const avgpool2dConfig: AvgPool2dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2, 2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        count_include_pad: countIncludePad,
        divisor_override: divisorOverride.trim() ? Number(divisorOverride) : undefined
      };
      
      return {
        type: 'avgpool2d',
        avgpool2d: avgpool2dConfig
      };
    } else if (selectedLayerType!.type === 'avgpool1d') {
      const avgpool1dConfig: AvgPool1dLayerConfigInput = {
        name: layerNameValue,
        kernel_size: parseArrayInput(poolKernelSize) || [2],
        stride: poolStride ? (parseArrayInput(poolStride) ?? undefined) : undefined,
        padding: poolPadding ? (parseArrayInput(poolPadding) ?? undefined) : undefined,
        count_include_pad: countIncludePad,
        divisor_override: divisorOverride.trim() ? Number(divisorOverride) : undefined
      };

      return {
        type: 'avgpool1d',
        avgpool1d: avgpool1dConfig
      };
    } else if (selectedLayerType!.type === 'batchnorm2d') {
      const batchnorm2dConfig: BatchNorm2dLayerConfigInput = {
        name: layerNameValue,
        num_features: parseInt(numFeatures, 10),
        eps: parseFloat(eps) || 1e-5,
        momentum: parseFloat(momentum) || 0.1,
        affine,
        track_running_status: trackRunningStatus
      };
      
      return {
        type: 'batchnorm2d',
        batchnorm2d: batchnorm2dConfig
      };
    } else if (selectedLayerType!.type === 'batchnorm1d') {
      const batchnorm1dConfig: BatchNorm1dLayerConfigInput = {
        name: layerNameValue,
        num_features: parseInt(numFeatures, 10),
        eps: parseFloat(eps) || 1e-5,
        momentum: parseFloat(momentum) || 0.1,
        affine,
        track_running_status: trackRunningStatus
      };
      
      return {
        type: 'batchnorm1d',
        batchnorm1d: batchnorm1dConfig
      };
    } else if (selectedLayerType!.type === 'flatten') {
      const flattenConfig: FlattenLayerConfigInput = {
        name: layerNameValue,
        start_dim: parseInt(startDim, 10) || 1,
        end_dim: parseInt(endDim, 10) || -1
      };
      
      return {
        type: 'flatten',
        flatten: flattenConfig
      };
    } else if (selectedLayerType!.type === 'dropout') {
      const dropoutConfig: DropoutLayerConfigInput = {
        name: layerNameValue,
        p: parseFloat(dropoutP) || 0.5
      };
      
      return {
        type: 'dropout',
        dropout: dropoutConfig
      };
    } else if (selectedLayerType!.type === 'elu') {
      const eluConfig: ELULayerConfigInput = {
        name: layerNameValue,
        alpha: parseFloat(alpha) || 1.0,
        inplace: eluInplace
      };
      
      return {
        type: 'elu',
        elu: eluConfig
      };
    } else if (selectedLayerType!.type === 'relu') {
      const reluConfig: ReLULayerConfigInput = {
        name: layerNameValue,
        inplace: reluInplace
      };
      
      return {
        type: 'relu',
        relu: reluConfig
      };
    } else if (selectedLayerType!.type === 'leakyrelu') {
      const leakyReluConfig: LeakyReLULayerConfigInput = {
        name: layerNameValue,
        negative_slope: parseFloat(negativeSlope) || 0.01,
        inplace: leakyReluInplace
      };
      
      return {
        type: 'leakyrelu',
        leakyrelu: leakyReluConfig
      };
    } else if (selectedLayerType!.type === 'sigmoid') {
      const sigmoidConfig: SigmoidLayerConfigInput = {
        name: layerNameValue
      };
      return { 
        type: 'sigmoid', 
        sigmoid: sigmoidConfig
      };
    } else if (selectedLayerType!.type === 'logsigmoid') {
      const logsigmoidConfig: LogSigmoidLayerConfigInput = {
        name: layerNameValue
      };
      return { 
        type: 'logsigmoid', 
        logsigmoid: logsigmoidConfig
      };
    } else if (selectedLayerType!.type === 'tanh') {
      const tanhConfig: TanhLayerConfigInput = {
        name: layerNameValue
      };
      return { 
        type: 'tanh', 
        tanh: tanhConfig
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

      console.log('Add to graph response:', response.data);

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
    } else if (layer.type === 'maxpool1d') {
      return {
        type: 'maxpool1d',
        maxpool1d: {
          name: layer.name,
          kernel_size: layer.kernel_size,
          stride: layer.stride,
          padding: layer.padding,
          dilation: layer.dilation,
          return_indices: layer.return_indices ?? false,
          ceil_mode: layer.ceil_mode ?? false
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
    } else if (config.type === 'maxpool2d' && config.maxpool2d) {
      layerName = config.maxpool2d.name ?? '';
      poolKernelSize = config.maxpool2d.kernel_size.join(',');
      poolStride = config.maxpool2d.stride ? config.maxpool2d.stride.join(',') : '';
      poolPadding = config.maxpool2d.padding ? config.maxpool2d.padding.join(',') : '';
      poolDilation = config.maxpool2d.dilation ? config.maxpool2d.dilation.join(',') : '';
      returnIndices = config.maxpool2d.return_indices ?? false;
      ceilMode = config.maxpool2d.ceil_mode ?? false;
    } else if (config.type === 'maxpool1d' && config.maxpool1d) {
      layerName = config.maxpool1d.name ?? '';
      poolKernelSize = config.maxpool1d.kernel_size.join(',');
      poolStride = config.maxpool1d.stride ? config.maxpool1d.stride.join(',') : '';
      poolPadding = config.maxpool1d.padding ? config.maxpool1d.padding.join(',') : '';
      poolDilation = config.maxpool1d.dilation ? config.maxpool1d.dilation.join(',') : '';
      returnIndices = config.maxpool1d.return_indices ?? false;
      ceilMode = config.maxpool1d.ceil_mode ?? false;
    } else if (config.type === 'avgpool2d' && config.avgpool2d) {
      layerName = config.avgpool2d.name ?? '';
      poolKernelSize = config.avgpool2d.kernel_size.join(',');
      poolStride = config.avgpool2d.stride ? config.avgpool2d.stride.join(',') : '';
      poolPadding = config.avgpool2d.padding ? config.avgpool2d.padding.join(',') : '';
      countIncludePad = config.avgpool2d.count_include_pad ?? false;
      divisorOverride = config.avgpool2d.divisor_override !== undefined && config.avgpool2d.divisor_override !== null
        ? config.avgpool2d.divisor_override.toString()
        : '';
    } else if (config.type === 'avgpool1d' && config.avgpool1d) {
      layerName = config.avgpool1d.name ?? '';
      poolKernelSize = config.avgpool1d.kernel_size.join(',');
      poolStride = config.avgpool1d.stride ? config.avgpool1d.stride.join(',') : '';
      poolPadding = config.avgpool1d.padding ? config.avgpool1d.padding.join(',') : '';
      countIncludePad = config.avgpool1d.count_include_pad ?? false;
      divisorOverride = config.avgpool1d.divisor_override !== undefined && config.avgpool1d.divisor_override !== null
        ? config.avgpool1d.divisor_override.toString()
        : '';
    } else if (config.type === 'batchnorm2d' && config.batchnorm2d) {
      layerName = config.batchnorm2d.name ?? '';
      numFeatures = config.batchnorm2d.num_features.toString();
      eps = config.batchnorm2d.eps?.toString() || '';
      momentum = config.batchnorm2d.momentum?.toString() || '0.1';
      affine = config.batchnorm2d.affine ?? true;
      trackRunningStatus = config.batchnorm2d.track_running_status ?? true;
    } else if (config.type === 'batchnorm1d' && config.batchnorm1d){
      layerName = config.batchnorm1d.name ?? '';
      numFeatures = config.batchnorm1d.num_features.toString();
      eps = config.batchnorm1d.eps?.toString() || '';
      momentum = config.batchnorm1d.momentum?.toString() || '0.1';
      affine = config.batchnorm1d.affine ?? true;
      trackRunningStatus = config.batchnorm1d.track_running_status ?? true;
    } else if (config.type === 'flatten' && config.flatten) {
      layerName = config.flatten.name ?? '';
      startDim = config.flatten.start_dim?.toString() || '1';
      endDim = config.flatten.end_dim?.toString() || '-1';
    } else if (config.type === 'dropout' && config.dropout) {
      layerName = config.dropout.name ?? '';
      dropoutP = config.dropout.p?.toString() || '0.5';
    } else if (config.type === 'elu' && config.elu) {
      layerName = config.elu.name ?? '';
      alpha = config.elu.alpha?.toString() || '1.0';
      eluInplace = config.elu.inplace ?? false;
    } else if (config.type === 'relu' && config.relu) {
      layerName = config.relu.name ?? '';
      reluInplace = config.relu.inplace ?? false;
    } else if (config.type === 'leakyrelu' && config.leakyrelu) {
      layerName = config.leakyrelu.name ?? '';
      negativeSlope = config.leakyrelu.negative_slope?.toString() || '0.01';
      leakyReluInplace = config.leakyrelu.inplace ?? false;
    } else if (config.type === 'sigmoid') {
      layerName = config.sigmoid?.name ?? '';
      // No additional fields for Sigmoid
    } else if (config.type === 'logsigmoid') {
      layerName = config.logsigmoid?.name ?? '';
      // No additional fields for LogSigmoid
    } else if (config.type === 'tanh') {
      layerName = config.tanh?.name ?? '';
      // No additional fields for Tanh
    } else {
      console.warn(`Unknown layer type: ${config.type}`);
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

      await validateGraphStructure();
      
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

      console.log('Validation response:', response.data);

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
          {node.layerConfig.linear?.name || node.layerConfig.conv2d?.name || node.layerConfig.conv1d?.name || 
          node.layerConfig.maxpool2d?.name || node.layerConfig.maxpool1d?.name || node.layerConfig.avgpool2d?.name || 
          node.layerConfig.avgpool1d?.name || node.layerConfig.batchnorm2d?.name || node.layerConfig.batchnorm1d?.name ||
          node.layerConfig.flatten?.name || node.layerConfig.dropout?.name || node.layerConfig.elu?.name ||
          node.layerConfig.relu?.name || node.layerConfig.leakyrelu?.name || node.layerConfig.sigmoid?.name ||
          node.layerConfig.logsigmoid?.name || node.layerConfig.tanh?.name || node.id}
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
        {:else if selectedLayerType.type === 'maxpool1d' || selectedLayerType.type === 'maxpool2d'}
          <label>
            Kernel Size:
            <input type="text" bind:value={poolKernelSize} placeholder= {selectedLayerType.type === 'maxpool2d' ? "e.g., 2,2" : "e.g., 2"}>
          </label>
          <label>
            Stride (optional):
            <input type="text" bind:value={poolStride} placeholder ={selectedLayerType.type === 'maxpool2d' ? "e.g., 2,2" : "e.g., 2"}>
          </label>
          <label>
            Padding (optional):
            <input type="text" bind:value={poolPadding} placeholder= {selectedLayerType.type === 'maxpool2d' ? "e.g., 0,0" : "e.g., 0"}>
          </label>
          <label>
            Dilation (optional):
            <input type="text" bind:value={poolDilation} placeholder= {selectedLayerType.type === 'maxpool2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            <input type="checkbox" bind:checked={returnIndices}>
            Return Indices
          </label>
          <label>
            <input type="checkbox" bind:checked={ceilMode}>
            Ceil Mode
          </label>
        {:else if selectedLayerType.type === 'avgpool1d' || selectedLayerType.type === 'avgpool2d'}
          <label>
            Kernel Size:
            <input type="text" bind:value={poolKernelSize} placeholder={selectedLayerType.type === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"}>
          </label>
          <label>
            Stride (optional):
            <input type="text" bind:value={poolStride} placeholder={selectedLayerType.type === 'avgpool2d' ? "e.g., 2,2" : "e.g., 2"}>
          </label>
          <label>
            Padding (optional):
            <input type="text" bind:value={poolPadding} placeholder={selectedLayerType.type === 'avgpool2d' ? "e.g., 0,0" : "e.g., 0"}>
          </label>
          <label>
            Count Include Pad:
            <input type="checkbox" bind:checked={countIncludePad}>
          </label>
          <label>
            Divisor Override:
            <input type="text" bind:value={divisorOverride} placeholder="Optional divisor override">
          </label>
        {:else if selectedLayerType.type === 'batchnorm1d' || selectedLayerType.type === 'batchnorm2d'}
          <label>
            Number of Features:
            <input type="text" bind:value={numFeatures} placeholder="Enter number of features">
          </label>
          <label>
            Epsilon:
            <input type="text" bind:value={eps} placeholder="Default: 1e-5">
          </label>
          <label>
            Momentum:
            <input type="text" bind:value={momentum} placeholder="Default: 0.1">
          </label>
          <label>
            <input type="checkbox" bind:checked={affine}>
            Affine
          </label>
          <label>
            <input type="checkbox" bind:checked={trackRunningStatus}>
            Track Running Status
          </label>
        {:else if selectedLayerType.type === 'flatten'}
          <label>
            Start Dimension:
            <input type="text" bind:value={startDim} placeholder="Default: 1">
          </label>
          <label>
            End Dimension:
            <input type="text" bind:value={endDim} placeholder="Default: -1">
          </label>
        {:else if selectedLayerType.type === 'dropout'}
          <label>
            Dropout Probability:
            <input type="text" bind:value={dropoutP} placeholder="Default: 0.5">
          </label>
        {:else if selectedLayerType.type === 'elu'}
          <label>
            Alpha:
            <input type="text" bind:value={alpha} placeholder="Default: 1.0">
          </label>
          <label>
            <input type="checkbox" bind:checked={eluInplace}>
            Inplace
          </label>
        {:else if selectedLayerType.type === 'relu'}
          <label>
            <input type="checkbox" bind:checked={reluInplace}>
            Inplace
          </label>
        {:else if selectedLayerType.type === 'leakyrelu'}
          <label>
            Negative Slope:
            <input type="text" bind:value={negativeSlope} placeholder="Default: 0.01">
          </label>
          <label>
            <input type="checkbox" bind:checked={leakyReluInplace}>
            Inplace
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
          {:else if selectedNode.layerConfig.type === 'maxpool1d' || selectedNode.layerConfig.type === 'maxpool2d'}
            <label>
              Kernel Size:
              <input type="text" bind:value={poolKernelSize}>
            </label>
            <label>
              Stride:
              <input type="text" bind:value={poolStride}>
            </label>
            <label>
              Padding:
              <input type="text" bind:value={poolPadding}>
            </label>
            <label>
              Dilation:
              <input type="text" bind:value={poolDilation}>
            </label>
            <label>
              <input type="checkbox" bind:checked={returnIndices}>
              Return Indices
            </label>
            <label>
              <input type="checkbox" bind:checked={ceilMode}>
              Ceil Mode
            </label>
          {:else if selectedNode.layerConfig.type === 'avgpool1d' || selectedNode.layerConfig.type === 'avgpool2d'}
            <label>
              Kernel Size:
              <input type="text" bind:value={poolKernelSize}>
            </label>
            <label>
              Stride:
              <input type="text" bind:value={poolStride}>
            </label>
            <label>
              Padding:
              <input type="text" bind:value={poolPadding}>
            </label>
            <label>
              Count Include Pad:
              <input type="checkbox" bind:checked={countIncludePad}>
            </label>
            <label>
              Divisor Override:
              <input type="text" bind:value={divisorOverride} placeholder="Optional divisor override">
            </label>
          {:else if selectedNode.layerConfig.type === 'batchnorm1d' || selectedNode.layerConfig.type === 'batchnorm2d'}
            <label>
              Number of Features:
              <input type="text" bind:value={numFeatures}>
            </label>
            <label>
              Epsilon:
              <input type="text" bind:value={eps} placeholder="Default: 1e-5">
            </label>
            <label>
              Momentum:
              <input type="text" bind:value={momentum} placeholder="Default: 0.1">
            </label>
            <label>
              <input type="checkbox" bind:checked={affine}>
              Affine
            </label>
            <label>
              <input type="checkbox" bind:checked={trackRunningStatus}>
              Track Running Status
            </label>
          {:else if selectedNode.layerConfig.type === 'flatten'}
            <label>
              Start Dimension:
              <input type="text" bind:value={startDim} placeholder="Default: 1">
            </label>
            <label>
              End Dimension:
              <input type="text" bind:value={endDim} placeholder="Default: -1">
            </label>
          {:else if selectedNode.layerConfig.type === 'dropout'}
            <label>
              Dropout Probability:
              <input type="text" bind:value={dropoutP} placeholder="Default: 0.5">
            </label>
          {:else if selectedNode.layerConfig.type === 'elu'}
            <label>
              Alpha:
              <input type="text" bind:value={alpha} placeholder="Default: 1.0">
            </label>
            <label>
              <input type="checkbox" bind:checked={eluInplace}>
              Inplace
            </label>
          {:else if selectedNode.layerConfig.type === 'relu'}
            <label>
              <input type="checkbox" bind:checked={reluInplace}>
              Inplace
            </label>
          {:else if selectedNode.layerConfig.type === 'leakyrelu'}
            <label>
              Negative Slope:
              <input type="text" bind:value={negativeSlope} placeholder="Default: 0.01">
            </label>
            <label>
              <input type="checkbox" bind:checked={leakyReluInplace}>
              Inplace
            </label>
          {:else if selectedNode.layerConfig.type === 'sigmoid'}
            <!-- No additional fields for Sigmoid -->
          {:else if selectedNode.layerConfig.type === 'logsigmoid'}
            <!-- No additional fields for LogSigmoid -->
          {:else if selectedNode.layerConfig.type === 'tanh'}
            <!-- No additional fields for Tanh -->
          {:else}
            <p>Unknown layer type: {selectedNode.layerConfig.type}</p>
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
              {node.layerConfig.linear?.name || node.layerConfig.conv2d?.name || node.layerConfig.conv1d?.name || 
                node.layerConfig.maxpool2d?.name || node.layerConfig.maxpool1d?.name || node.layerConfig.avgpool2d?.name || 
                node.layerConfig.avgpool1d?.name || node.layerConfig.batchnorm2d?.name || node.layerConfig.batchnorm1d?.name ||
                node.layerConfig.flatten?.name || node.layerConfig.dropout?.name || node.layerConfig.elu?.name ||
                node.layerConfig.relu?.name || node.layerConfig.leakyrelu?.name || node.layerConfig.sigmoid?.name ||
                node.layerConfig.logsigmoid?.name || node.layerConfig.tanh?.name || node.id} 
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
            <div class="build-summary">
              <div class="summary-item">
                <span class="label">Layers:</span>
                <span class="value">{buildResult.module_graph.layers.length}</span>
              </div>
              <div class="summary-item">
                <span class="label">Connections:</span>
                <span class="value">{buildResult.module_graph.edges.length}</span>
              </div>
            </div>
            
            <!-- Graph Validation Results -->
            {#if graphValidationResult}
              <div class="graph-validation">
                <h5>🔍 Graph Validation</h5>
                
                {#if graphValidationResult.status && graphValidationResult.status.length > 0}
                  <!-- Check if there are validation errors -->
                  {#if graphValidationResult.status.some((s: any) => s.message)}
                    <div class="validation-error">
                      <div class="status-header">
                        <span class="status-icon">⚠️</span>
                        <strong>Validation Issues Found</strong>
                      </div>
                      
                      {#each graphValidationResult.status as status, index}
                        {#if status.message}
                          <div class="error-item">
                            <div class="error-number">Issue #{index + 1}</div>
                            <div class="error-details">
                              <p class="error-message">{status.message}</p>
                              
                              {#if status.out_dimension && status.out_dimension.length > 0}
                                <div class="dimension-info">
                                  <span class="dim-label">Output:</span>
                                  <code class="dimension">[{status.out_dimension.join(', ')}]</code>
                                </div>
                              {/if}
                              
                              {#if status.required_in_dimension && status.required_in_dimension.length > 0}
                                <div class="dimension-info">
                                  <span class="dim-label">Required Input:</span>
                                  <code class="dimension">[{status.required_in_dimension.join(', ')}]</code>
                                </div>
                              {/if}
                            </div>
                          </div>
                        {/if}
                      {/each}
                    </div>
                  {:else}
                    <div class="validation-success">
                      <div class="status-header">
                        <span class="status-icon">✅</span>
                        <strong>Graph is Valid!</strong>
                      </div>
                      <p>All layer dimensions match correctly</p>
                    </div>
                  {/if}
                  
          
                {:else}
                  <div class="no-validation">
                    <p>No validation data available</p>
                  </div>
                {/if}
              </div>
            {/if}
            
            <!-- Layer Execution Order -->
            {#if buildResult.module_graph.sorted && buildResult.module_graph.layers.length > 0}
              <div class="layer-order">
                <h5>🔄 Layer Execution Order</h5>
                <div class="layer-list">
                  {#each buildResult.module_graph.layers as layer, index}
                    <div class="layer-item">
                      <div class="layer-info">
                        <span class="layer-name">{layer.name}</span>
                        <span class="layer-type">({layer.type})</span>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>