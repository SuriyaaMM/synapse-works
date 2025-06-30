<script lang="ts">
  import { goto } from '$app/navigation';
  import client from '$lib/apolloClient';
  import { CREATE_MODEL, LOAD_MODEL } from '$lib/mutations'; // Updated import
  import { GET_MODELS } from '$lib/queries';
  import { onMount } from 'svelte';
  import type { Model, CreateModelArgs } from '../../../../source/types';

  import './create-model.css';

  // Form state
  let modelName: string = '';
  let loading: boolean = false;
  let error: string | null = null;
  let savedModels: Model[] = [];
  let availableModels: Model[] = [];
  let loadingSavedModels: boolean = false;
  let loadingAvailableModels: boolean = false;

  // Create new model and navigate to layer configuration
  async function createModel(): Promise<void> {
    if (!modelName.trim()) {
      error = 'Please enter a model name';
      return;
    }

    loading = true;
    error = null;

    try {
      const createModelArgs: CreateModelArgs = { 
        name: modelName.trim() 
      };

      const res = await client.mutate({
        mutation: CREATE_MODEL,
        variables: createModelArgs
      });

      const model: Model | undefined = res.data?.createModel;
      if (!model?.id) {
        throw new Error('Model creation failed - no ID returned');
      }

      // Reload models after creating a new one
      await loadSavedModels();

      // Navigate to layer configuration page
      await goto(`/model/${model.id}/layer-config`);
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  // Load a specific model using LOAD_UNET_MODEL mutation
  async function loadSpecificModel(modelId: string): Promise<Model | null> {
    try {
      const res = await client.mutate({
        mutation: LOAD_MODEL,
        variables: { modelId }
      });
      return res.data?.loadModel || null;
    } catch (err) {
      console.error('Error loading specific model:', err);
      return null;
    }
  }

  // Load saved models - now using GET_MODELS since LOAD_MODEL doesn't return all models
  async function loadSavedModels(): Promise<void> {
    loadingSavedModels = true;
    try {
      // Since LOAD_UNET_MODEL requires a specific modelId, we'll use GET_MODELS
      // to get the list of available models, then optionally load specific ones
      const res = await client.query({
        query: GET_MODELS,
        fetchPolicy: 'network-only'
      });
      savedModels = res.data?.getModels || [];
      
      // If you need to load specific model details using LOAD_UNET_MODEL,
      // you can do so for each model:
      // for (const model of savedModels) {
      //   const detailedModel = await loadSpecificModel(model.id);
      //   if (detailedModel) {
      //     // Update the model with detailed information
      //   }
      // }
      
    } catch (err) {
      console.error('Error loading saved models:', err);
    } finally {
      loadingSavedModels = false;
    }
  }

  // Load available models using GET_MODELS query
  async function loadAvailableModels(): Promise<void> {
    loadingAvailableModels = true;
    try {
      const res = await client.query({
        query: GET_MODELS,
        fetchPolicy: 'network-only' // Always fetch fresh data
      });
      availableModels = res.data?.getModels || [];
    } catch (err) {
      console.error('Error loading available models:', err);
    } finally {
      loadingAvailableModels = false;
    }
  }

  // Load all models
  async function loadAllModels(): Promise<void> {
    await Promise.all([loadSavedModels(), loadAvailableModels()]);
  }

  // Load models on component mount
  onMount(() => {
    loadAllModels();
  });
</script>

<div class="full-page">
  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <img src="/brain.png" alt="Brain Icon" class="header-icon" />
      <h1 class="header-title">Synapse Works</h1>
    </div>

    <div class="profile-icon">👤</div>
  </div>

  <!-- Main content -->
  <div class="main-content">
    <!-- Left Panel: Create Model -->
    <div class="left-panel">
      <h2 class="panel-title">Create Model</h2>

      <form on:submit|preventDefault={createModel} class="form">
        <input
          type="text"
          bind:value={modelName}
          placeholder="Enter the model name"
          required
          class="form-input"
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading || !modelName.trim()}
          class="form-button"
        >
          {loading ? 'Creating...' : 'Create Model'}
        </button>

        {#if error}
          <div class="form-error">{error}</div>
        {/if}
      </form>
    </div>

    <!-- Right Panel: Models -->
    <div class="right-panel">
      <!-- Your Models Section -->
      <div class="model-section">
        <div class="section-header">
          <h2 class="section-title">Your Models</h2>
          <button
            on:click={loadSavedModels}
            disabled={loadingSavedModels}
            class="refresh-button"
          >
            {loadingSavedModels ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {#if loadingSavedModels}
          <div class="loading-section">
            <div class="loading-text">Loading saved models...</div>
          </div>
        {:else if savedModels.length === 0}
          <p class="empty-text">No saved models found. Create your first model to get started!</p>
        {:else}
          <div class="model-list">
            {#each savedModels as model}
              <div class="model-card">
                <div class="model-card-header">
                  <div class="model-details">
                    <h3 class="model-name">{model.name}</h3>
                    <p class="model-id">ID: {model.id}</p>

                    {#if model.layers_config && model.layers_config.length > 0}
                      <div class="model-layers">
                        <p class="model-layer-title">Layers ({model.layers_config.length}):</p>
                        <div class="model-layer-list">
                          {#each model.layers_config as layer}
                            <div class="layer-item">
                              <span class="layer-type">{layer.type}</span>
                              <span class="dot">•</span>
                              <span class="layer-name">{layer.name}</span>
                              <span class="layer-id">(ID: {layer.id})</span>
                            </div>
                          {/each}
                        </div>
                      </div>
                    {:else}
                      <p class="model-no-layer">No layers configured</p>
                    {/if}

                    {#if model.dataset_config}
                      <p class="model-dataset">Dataset: {model.dataset_config.name}</p>
                    {/if}
                  </div>

                  <div class="model-actions">
                    <button
                      on:click={() => goto(`/model/${model.id}/layer-config`)}
                      class="view-edit-button"
                    >
                      View/Edit
                    </button>
                    <!-- Add button to load specific model details -->
                    <button
                      on:click={() => loadSpecificModel(model.id)}
                      class="load-button"
                    >
                      Load Details
                    </button>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Available Models Section -->
      <div class="model-section available-section">
        <div class="section-header">
          <h2 class="section-title">Available Models</h2>
          <button
            on:click={loadAvailableModels}
            disabled={loadingAvailableModels}
            class="refresh-button"
          >
            {loadingAvailableModels ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {#if loadingAvailableModels}
          <div class="loading-section">
            <div class="loading-text">Loading available models...</div>
          </div>
        {:else if availableModels.length === 0}
          <p class="empty-text">No available models found.</p>
        {:else}
          <div class="model-list">
            {#each availableModels as model}
              <div class="available-card">
                <div class="model-card-header">
                  <div class="model-details">
                    <h3 class="model-name">{model.name}</h3>
                    <p class="model-id">ID: {model.id}</p>

                    {#if model.layers_config && model.layers_config.length > 0}
                      <div class="model-layers">
                        <p class="model-layer-title">Layers ({model.layers_config.length}):</p>
                        <div class="model-layer-list">
                          {#each model.layers_config as layer}
                            <div class="layer-item">
                              <span class="layer-type available-layer">{layer.type}</span>
                              <span class="dot">•</span>
                              <span class="layer-name">{layer.name}</span>
                              <span class="layer-id">(ID: {layer.id})</span>
                            </div>
                          {/each}
                        </div>
                      </div>
                    {:else}
                      <p class="model-no-layer">No layers configured</p>
                    {/if}

                    {#if model.dataset_config}
                      <p class="model-dataset">Dataset: {model.dataset_config.name}</p>
                    {/if}
                  </div>

                  <div class="model-actions">
                    <button
                      on:click={() => goto(`/model/${model.id}/layer-config`)}
                      class="view-edit-button"
                    >
                      View/Edit
                    </button>
                    <!-- Add button to load specific model details -->
                    <button
                      on:click={() => loadSpecificModel(model.id)}
                      class="load-button"
                    >
                      Load Details
                    </button>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>