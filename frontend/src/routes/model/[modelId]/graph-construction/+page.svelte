<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { page } from '$app/stores';
  import { SvelteFlow } from '@xyflow/svelte';
  import type { LayerConfigInput } from '../../../../../../source/types/layerTypes';
  import LayerForm from './LayerForm.svelte';
  import '@xyflow/svelte/dist/style.css';
  import './graph-construction.css';

  // Import our organized modules
  import { graphStore } from './graphStore.svelte';
  import { createStorageService } from './storage';
  import { createEventHandlers } from './eventHandlers';
  import { createEventDispatcher } from 'svelte';
  import { createGraphMutations } from './graphMutations';
  import { fetchModelDetails } from '../modelDetails';
  import { layerTypes } from './types';
  import type { FlowNode } from './types';

  // Component state
  let modelId: string | null = null;
  let showNodeDetails = $state(false);
  let detailsNode: FlowNode | null = $state(null);

  let showDropdown = $state(false);

  let currentModal = $state<'nodeDetails' | 'layerForm' | 'build-graph' | null>(null);

  const dispatch = createEventDispatcher();

  const storageService = $derived(() => 
    modelId ? createStorageService(modelId) : null
  );

const eventHandlers = $derived(() => {
    if (!storageService) return null;
    
    return createEventHandlers(
        mutations()?.createConnection || (() => {}),
        showNodeDetailsView,
        storageService()?.saveStateToStorage || (() => {})
    );
});

const mutations = $derived(() => {
    if (!storageService) return null;
    
    return createGraphMutations(
        storageService()?.saveStateToStorage || (() => {}),
        (eventName, detail) => dispatch(eventName, detail)
    );
});


  // Extract modelId from URL
  $effect(() => {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    modelId = (modelIndex !== -1 && modelIndex + 1 < pathParts.length) ? pathParts[modelIndex + 1] : null;
  });

  // Load state when model changes
  $effect(() => {
  if (modelId && storageService) {
    const service = storageService();
    setTimeout(() => {
      service?.loadStateFromStorage();
    }, 100);
  }
});

$effect(() => {
  if (modelId && storageService) {
    const service = storageService();
    // Save state every 30 seconds as a backup
    const interval = setInterval(() => {
      service?.saveStateToStorage();
    }, 30000);

    return () => clearInterval(interval);
  }
});

function forceeSavePositions() {
  if (storageService) {
    const service = storageService();
    service?.saveStateToStorage();
  }
}

  function showNodeDetailsView(node: FlowNode) {
    detailsNode = node;
    currentModal = 'nodeDetails';
  }

  function closeModal() {
    graphStore.setSelectedLayerType(null);
    detailsNode = null;
    currentModal = null;
  }
  function selectLayerType(layerType: typeof layerTypes[number]) {
    graphStore.setSelectedLayerType(layerType);
    graphStore.setSelectedNode(null);
    graphStore.setSelectedEdge(null);
    currentModal = 'layerForm';
    showDropdown = false;
  }

  function clearSelection() {
    graphStore.clearSelection();
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Escape' && showNodeDetails) {
      closeModal();
      return;
    }
    
    if ((event.key === 'Delete' || event.key === 'Backspace') && !graphStore.loading) {
      console.log('Delete key pressed');
      console.log('Current selectedEdge:', graphStore.selectedEdge);
      console.log('Current selectedNode:', graphStore.selectedNode);
      
      if (graphStore.selectedNode) {
        mutations()?.handleDeleteNode();
      } else if (graphStore.selectedEdge) {
        console.log('Selected edge:', graphStore.selectedEdge);
        mutations()?.handleDeleteEdge();
      } else {
        console.log('No node or edge selected');
      }
    }
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

  onMount(() => {
    fetchModelDetails();
    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  });

  function buildGraph() {
  console.log('🔍 Build Graph clicked');
  console.log('Current modal before:', currentModal);
  console.log('Current buildResult:', graphStore.buildResult);
  console.log('Current graphValidationResult:', graphStore.graphValidationResult);
  
  mutations()?.buildModuleGraph();
  
  currentModal = 'build-graph';
  
  console.log('Current modal after:', currentModal);
  
  setTimeout(() => {
    console.log('Modal state after timeout:', currentModal);
    console.log('Build result after timeout:', graphStore.buildResult);
  }, 100);
}
</script>

<div class="graph-editor">
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

  <div class="dropdown">
    <button class="dropdown-button" onclick={() => showDropdown = !showDropdown}>
      {graphStore.selectedLayerType?.label || 'Select Layer Type'}
    </button>

    {#if showDropdown}
      <div class="dropdown-content">
        {#each layerTypes as layerType}
          <button 
            class:selected={graphStore.selectedLayerType?.type === layerType.type}
            style="border-left: 4px solid {layerType.color}"
            onclick={() => selectLayerType(layerType)}
          >
            {layerType.label}
          </button>
        {/each}
      </div>
    {/if}
  </div>
    <!-- Build Graph Button -->
  <div class="build-graph-controls">
    <button 
      class="build-graph-button" 
      onclick={buildGraph}
      disabled={graphStore.loading}
    >
      {graphStore.loading ? 'Building...' : 'Build & Sort Graph'}
    </button>
    {#if graphStore.error}
      <div class="error-message">{graphStore.error}</div>
    {/if}
  </div>

    <div style="width: 100%; height: 100vh;">
      <SvelteFlow 
    nodes={graphStore.nodes}
    edges={graphStore.edges}
    onnodeclick={eventHandlers()?.onNodeClick}
    onedgeclick={eventHandlers()?.onEdgeClick}
    onconnect={eventHandlers()?.onConnect}
    onnodeschange={eventHandlers()?.onNodesChange}
    onnodedrag={eventHandlers()?.onNodeDrag}
    onnodedragstop={eventHandlers()?.onNodeDragStop}
    fitView
    nodeTypes={{}}
    edgeTypes={{}}
  >
        <!-- Custom Node Template -->
        <template slot="node" let:node>
          <div
            class="custom-node {graphStore.selectedNode?.id === node.id ? 'selected' : ''}"
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
    {#if currentModal === 'nodeDetails' && detailsNode}
      <div
        class="node-details-overlay"
        role="button"
        tabindex="0"
        aria-label="Close node details"
        onclick={closeModal}
        onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') { closeModal(); } }}
      >
        <div
          class="node-details-modal"
          role="dialog"
          aria-modal="true"
          tabindex="0"
          onclick={(e) => e.stopPropagation()}
          onkeydown={(e) => { if (e.key === 'Escape') { closeModal(); } }}
        >
          <div class="details-header" style="background-color: {detailsNode.data.color}">
            <h3>{detailsNode.data.label}</h3>
            <button class="close-button" onclick={closeModal}>×</button>
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
    {:else if (currentModal === 'layerForm' && graphStore.selectedLayerType)}
      <div
        class="layer-form-overlay"
        role="button"
        tabindex="0"
        aria-label="Close layer form"
        onclick={closeModal}
        onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') { closeModal(); } }}
      >
        <div
          class="layer-form-modal"
          role="dialog"
          aria-modal="true"
          tabindex="0"
          onclick={(e) => e.stopPropagation()}
          onkeydown={(e) => { if (e.key === 'Escape') { closeModal(); } }}
        >
          <div class="details-header" style="background-color: #3b82f6;">
            <h3>Add {graphStore.selectedLayerType.label}</h3>
            <button class="close-button" onclick={closeModal}>×</button>
          </div>

          <div class="details-content layer-form-content">
            <LayerForm 
              selectedLayerType={graphStore.selectedLayerType}
              nodes={graphStore.nodes}
              loading={graphStore.loading}
              on:addLayer={(e) => { void (mutations()?.handleAddLayer?.(e)); }}
            />
          </div>

          <div class="details-footer">
            <span class="hint">Press ESC to close</span>
          </div>
        </div>
      </div>
    {:else if currentModal === 'build-graph'}
      <div
        class="build-graph-overlay"
        role="button"
        tabindex="0"
        aria-label="Close build-graph"
        onclick={closeModal}
        onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') { closeModal(); } }}
      >
        <div
          class="build-graph-modal"
          role="dialog"
          aria-modal="true"
          tabindex="0"
          onclick={(e) => e.stopPropagation()}
          onkeydown={(e) => { if (e.key === 'Escape') { closeModal(); } }}
        >
          <div class="details-header" style="background-color: #6366f1;">
            <h3>Build Summary</h3>
            <button class="close-button" onclick={closeModal}>×</button>
          </div>

          <div class="details-content">
            {#if graphStore.loading}
              <div class="loading-content">
                <h4>Building graph...</h4>
                <p>Please wait while the graph is being constructed.</p>
              </div>
            {:else if graphStore.buildResult}
              <div class="build-result">
                <h4>Built Graph Successfully</h4>
                <div class="build-summary">
                  <div class="summary-item">
                    <span class="label">Layers:</span>
                    <span class="value">{graphStore.buildResult.module_graph?.layers?.length || 0}</span>
                  </div>
                  <div class="summary-item">
                    <span class="label">Connections:</span>
                    <span class="value">{graphStore.buildResult.module_graph?.edges?.length || 0}</span>
                  </div>
                </div>

                <div class="graph-validation">
                  <h5>🔍 Graph Validation</h5>

                  {#if graphStore.graphValidationResult?.status && graphStore.graphValidationResult.status.length > 0}
                    {#if graphStore.graphValidationResult.status.some((s: any) => s.message)}
                      <div class="validation-error">
                        <div class="status-header">
                          <span class="status-icon">⚠️</span>
                          <strong>Validation Issues Found</strong>
                        </div>

                        {#each graphStore.graphValidationResult.status as status, index}
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
                      <div class="status-header">
                        <span class="status-icon">ℹ️</span>
                        <strong>No Validation Data</strong>
                      </div>
                      <p>Graph validation results are not available yet.</p>
                    </div>
                  {/if}
                </div>

                {#if graphStore.buildResult.module_graph?.sorted && graphStore.buildResult.module_graph.layers?.length > 0}
                  <div class="layer-order">
                    <h5>🔄 Layer Execution Order</h5>
                    <div class="layer-list">
                      {#each graphStore.buildResult.module_graph.layers as layer}
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
            {:else}
              <div class="error-content">
                <h4>No Build Result Available</h4>
                <p>The graph build process did not return any results.</p>
                {#if graphStore.error}
                  <div class="error-message">{graphStore.error}</div>
                {/if}
              </div>
            {/if}
          </div>
          
          <div class="details-footer">
            <span class="hint">Press ESC to close</span>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>