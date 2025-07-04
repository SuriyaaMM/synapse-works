<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { LayerConfigInput, LinearLayerConfigInput, 
                Conv2dLayerConfigInput, Conv1dLayerConfigInput, 
                MaxPool2dLayerConfigInput, MaxPool1dLayerConfigInput,
                AvgPool2dLayerConfigInput, AvgPool1dLayerConfigInput, 
                BatchNorm2dLayerConfigInput, BatchNorm1dLayerConfigInput,
                FlattenLayerConfigInput, DropoutLayerConfigInput,
                ELULayerConfigInput, ReLULayerConfigInput,
                LeakyReLULayerConfigInput, SigmoidLayerConfigInput, 
                LogSigmoidLayerConfigInput, TanhLayerConfigInput,
                ConvTranspose2dLayerConfigInput, CatLayerConfigInput,
                Dropout2dLayerConfigInput} from '../../../../../../source/types/layerTypes';

  // Props
  export let selectedLayerType: any = null;
  export let nodes: any[] = [];
  export const loading: boolean = false;
  export const layerTypes: any[] = [];
  export let buildResult: any = null;
  export let graphValidationResult: any = null;

  // Events
  const dispatch = createEventDispatcher<{
    addLayer: { layerConfig: LayerConfigInput };
    deleteNode: void;
    deleteEdge: void;
    updateNode: { nodeId: string; layerConfig: LayerConfigInput };
  }>();

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

  let outputPadding = '';
  let catDimension = '';

  let formError = '';


  function resetFormFields() {
    formError = '';
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
    
    if (numbers.some(n => isNaN(n) || n < 0)) return null;
    
    return numbers;
  }

  function validateForm(): string | null {
    console.log("1. layertype:", selectedLayerType);
    if (!selectedLayerType) return 'Layer type is required';
    
    const t = selectedLayerType.type;
    const validateNum = (val: string, name: string, min = 0, max = Infinity) => {
        const n = Number(val);
        if (!val || isNaN(n) || n <= min || n > max) 
            return `${name} must be ${min === 0 ? 'a positive number' : `between ${min} and ${max}`}`;
        return null;
    };
    
    const validateKernel = (size: string, dims: number, name = 'Kernel size') => {
        if (!size?.trim()) return `${name} is required`;
        const arr = parseArrayInput(size);
        if (!arr || arr.length !== dims) 
            return `${name} must be ${dims === 1 ? 'a number' : `an array of exactly ${dims} numbers`} (e.g., "${dims === 1 ? '3' : '3,3'}")`;
        return null;
    };
    
    if (t === 'linear') {
        return validateNum(inFeatures, 'Input features') || validateNum(outFeatures, 'Output features');
    }
    
    if (t === 'conv2d' || t === 'conv1d' || t === 'convtranspose2d') {
      const dims = t === 'conv2d' || t === 'convtranspose2d' ? 2 : 1;

      const baseValidation =
        validateNum(inChannels, 'Input channels') ||
        validateNum(outChannels, 'Output channels') ||
        validateKernel(kernelSize, dims) ||
        validateNum(groups, 'Groups');

      if (baseValidation) return baseValidation;

      // Validate output_padding only for convtranspose2d
      if (t === 'convtranspose2d' && outputPadding.trim()) {
        const arr = parseArrayInput(outputPadding);
        if (!arr || arr.length !== 2 || arr.some(n => isNaN(n) || n < 0)) {
          return 'Output padding must be an array of exactly 2 non-negative numbers (e.g., "0,0")';
        }
      }

      return null;
    }

    if (t.includes('pool')) {
        const dims = t.includes('2d') ? 2 : 1;
        return validateKernel(poolKernelSize, dims);
    }
    
    if (t.includes('batchnorm')) {
        return validateNum(numFeatures, 'Number of features') || 
               validateNum(eps, 'Eps') || 
               validateNum(momentum, 'Momentum', 0, 1);
    }
    
    if (t === 'flatten') {
        return validateNum(startDim, 'Start dimension', -Infinity) || 
               validateNum(endDim, 'End dimension', -Infinity);
    }
    
    if (t === 'dropout' || t === 'dropout2d') return validateNum(dropoutP, 'Dropout probability', 0, 1);
    if (t === 'elu') return validateNum(alpha, 'Alpha');
    if (t === 'leakyrelu') return isNaN(Number(negativeSlope)) ? 'Negative slope must be a valid number' : null;
    
    return null;
  }

  function createLayerConfig(): LayerConfigInput {
    const layerNameValue = layerName.trim() || `${selectedLayerType!.type}Layer`;
    console.log("2. layertype:", selectedLayerType);
    
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
    } else if (selectedLayerType!.type === 'convtranspose2d') {
      const convTranspose2dConfig: ConvTranspose2dLayerConfigInput = {
        name: layerNameValue,
        in_channels: parseInt(inChannels, 10),
        out_channels: parseInt(outChannels, 10),
        kernel_size: parseArrayInput(kernelSize) || [3, 3],
        bias: convBias
      };
      
      if (stride.trim()) {
        const strideArray = parseArrayInput(stride);
        if (strideArray) {
          convTranspose2dConfig.stride = strideArray;
        }
      }
      if (padding.trim()) {
        const paddingArray = parseArrayInput(padding);
        if (paddingArray) {
          convTranspose2dConfig.padding = paddingArray;
        }
      }
      if (dilation.trim()) {
        const dilationArray = parseArrayInput(dilation);
        if (dilationArray) {
          convTranspose2dConfig.dilation = dilationArray;
        }
      }
      if (groups.trim() && groups !== '1') {
        convTranspose2dConfig.groups = [parseInt(groups, 10)];
      }
      if (outputPadding.trim()) {
        const outputPaddingArray = parseArrayInput(outputPadding);
        if (outputPaddingArray) {
          convTranspose2dConfig.output_padding = outputPaddingArray;
        }
      }
      
      return {
        type: 'convtranspose2d',
        convtranspose2d: convTranspose2dConfig
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
    } else if (selectedLayerType!.type === 'dropout2d') {
      const dropout2dConfig: Dropout2dLayerConfigInput = {
        name: layerNameValue,
        p: parseFloat(dropoutP) || 0.5
      };
      
      return {
        type: 'dropout2d',
        dropout2d: dropout2dConfig
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
    } else if (selectedLayerType!.type === 'cat') {
      const catConfig : CatLayerConfigInput= {
        name: layerNameValue,
        dimension: parseInt(catDimension, 10) || 1
      };
      return { 
        type: 'cat', 
        cat: catConfig
      };
    }

    throw new Error(`Unsupported layer type: ${selectedLayerType!.type}`);
  }

  function handleAddLayer() {
    const validationError = validateForm();
    if (validationError) {
      formError = validationError;
      return;
    }
    
    formError = ''; 
    const layerConfig = createLayerConfig();
    console.log("3. layertype:", layerConfig);
    dispatch('addLayer', { layerConfig });
    resetFormFields();
  }
</script>

<div class="sidebar right">
    {#if selectedLayerType}
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
        {:else if selectedLayerType.type === 'conv2d' || selectedLayerType.type === 'conv1d' || selectedLayerType.type === 'convtranspose2d'}
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
            <input type="text" bind:value={kernelSize} placeholder={selectedLayerType.type === 'conv2d' || selectedLayerType.type === 'convtranspose2d' ? "e.g., 3,3" : "e.g., 3"}>
          </label>
          <label>
            Stride (optional):
            <input type="text" bind:value={stride} placeholder={selectedLayerType.type === 'conv2d' || selectedLayerType.type === 'convtranspose2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            Padding (optional):
            <input type="text" bind:value={padding} placeholder={selectedLayerType.type === 'conv2d' || selectedLayerType.type === 'convtranspose2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            Dilation (optional):
            <input type="text" bind:value={dilation} placeholder={selectedLayerType.type === 'conv2d' || selectedLayerType.type === 'convtranspose2d' ? "e.g., 1,1" : "e.g., 1"}>
          </label>
          <label>
            Groups:
            <input type="text" bind:value={groups} placeholder="Default: 1">
          </label>

          {#if selectedLayerType.type !== 'convtranspose2d'}
            <label>
              Padding Mode (optional):
              <select bind:value={paddingMode}>
                <option value="zeros">zeros</option>
                <option value="reflect">reflect</option>
                <option value="replicate">replicate</option>
                <option value="circular">circular</option>
              </select>
            </label>
          {/if}

          {#if selectedLayerType.type === 'convtranspose2d'}
            <label>
              Output Padding (optional):
              <input type="text" bind:value={outputPadding} placeholder="e.g., 0,0">
            </label>
          {/if}

          <label>
            <input type="checkbox" bind:checked={convBias}>
            Bias (optional)
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
            Return Indices (optional)
          </label>
          <label>
            <input type="checkbox" bind:checked={ceilMode}>
            Ceil Mode (optional)
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
            Count Include Pad (optional):
            <input type="checkbox" bind:checked={countIncludePad}>
          </label>
          <label>
            Divisor Override (optional):
            <input type="text" bind:value={divisorOverride} placeholder="Optional divisor override">
          </label>
        {:else if selectedLayerType.type === 'batchnorm1d' || selectedLayerType.type === 'batchnorm2d'}
          <label>
            Number of Features :
            <input type="text" bind:value={numFeatures} placeholder="Enter number of features">
          </label>
          <label>
            Epsilon (optional):
            <input type="text" bind:value={eps} placeholder="Default: 1e-5">
          </label>
          <label>
            Momentum (optional):
            <input type="text" bind:value={momentum} placeholder="Default: 0.1">
          </label>
          <label>
            <input type="checkbox" bind:checked={affine}>
            Affine (optional)
          </label>
          <label>
            <input type="checkbox" bind:checked={trackRunningStatus}>
            Track Running Status (optional)
          </label>
        {:else if selectedLayerType.type === 'flatten'}
          <label>
            Start Dimension (optional):
            <input type="text" bind:value={startDim} placeholder="Default: 1">
          </label>
          <label>
            End Dimension (optional):
            <input type="text" bind:value={endDim} placeholder="Default: -1">
          </label>
        {:else if selectedLayerType.type === 'dropout' || selectedLayerType.type === 'dropout2d'}
          <label>
            Dropout Probability (optional):
            <input type="text" bind:value={dropoutP} placeholder="Default: 0.5">
          </label>
        {:else if selectedLayerType.type === 'elu'}
          <label>
            Alpha (optional):
            <input type="text" bind:value={alpha} placeholder="Default: 1.0">
          </label>
          <label>
            <input type="checkbox" bind:checked={eluInplace}>
            Inplace (optional)
          </label>
        {:else if selectedLayerType.type === 'relu'}
          <label>
            <input type="checkbox" bind:checked={reluInplace}>
            Inplace (optional)
          </label>
        {:else if selectedLayerType.type === 'leakyrelu'}
          <label>
            Negative Slope (optional):
            <input type="text" bind:value={negativeSlope} placeholder="Default: 0.01">
          </label>
          <label>
            <input type="checkbox" bind:checked={leakyReluInplace}>
            Inplace (optional)
          </label>
        {:else if selectedLayerType.type === 'cat'}
          <label>
            Concatenation Dimension (optional):
            <input type="text" bind:value={catDimension} placeholder="e.g., 1">
          </label>
        {/if}
        {#if formError}
          <div class="error-message">{formError}</div>
        {/if}
        
        <button class="add-button" on:click={handleAddLayer}>Add to Graph</button>
      </div>
    {/if}
  </div>