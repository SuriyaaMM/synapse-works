<script lang="ts">
  import { goto } from '$app/navigation';
  import client from '$lib/apolloClient';
  import { CREATE_MODEL, LOAD_MODEL } from '$lib/mutations';
  import type { Model } from '../../../../source/types/modelTypes';
  import type { CreateModelArgs } from '../../../../source/types/argTypes';

  import './create-model.css';

  // Form state
  let modelName: string = '';
  let loading: boolean = false;
  let error: string | null = null;
  let fileInput: HTMLInputElement;

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

      // Navigate to layer configuration page
      await goto(`/model/${model.id}/dataset-config`);
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  function showLoadModelDialog(): void {
    if (fileInput) {
      fileInput.click();
    }
  }

  async function handleFileLoad(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    
    if (!file) return;

    loading = true;
    error = null;

    try {
      const fileContent = await file.text();
      const modelData = JSON.parse(fileContent);
      
      console.log('Loaded model data:', modelData);
      
      if (!modelData.id) {
        throw new Error('Invalid model file: Missing model ID');
      }

      const loadedModel = await loadSpecificModel(modelData.id);
      
      if (loadedModel) {
        await goto(`/model/${modelData.id}/dataset-config`);
      } else {
        console.warn('Backend model loading failed, but navigating with file data');
        await goto(`/model/${modelData.id}/layer-config`);
      }
      
    } catch (err) {
      console.error('Error loading model file:', err);
      error = err instanceof Error ? err.message : 'Error loading model file. Please check the file format.';
    } finally {
      loading = false;
      input.value = '';
    }
  }

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
</script>

<!-- Hidden file input for loading model files -->
<input
  bind:this={fileInput}
  type="file"
  accept=".json,.model,.txt"
  on:change={handleFileLoad}
  style="display: none;"
/>

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

    <!-- Right Panel: Load Model -->
    <div class="right-panel">
      <div class="model-section">
        <div class="section-header">
          <h2 class="section-title">Load Saved Model</h2>
        </div>

        <div class="load-model-section">
          <p class="load-description">
            Click the button below to load a previously saved model file from your computer.
          </p>
          
          <button
            on:click={showLoadModelDialog}
            disabled={loading}
            class="load-file-button"
          >
            {loading ? '⏳ Loading...' : '📂 Choose Model File'}
          </button>

          {#if error}
            <div class="load-error">{error}</div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>