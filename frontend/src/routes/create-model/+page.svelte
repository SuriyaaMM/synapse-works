<script lang="ts">
  import { goto } from '$app/navigation';
  import client from '$lib/apolloClient';
  import { CREATE_MODEL } from '$lib/mutations';
  import { GET_MODELS } from '$lib/queries';
  import { onMount } from 'svelte';
  
  import type { Model, CreateModelArgs } from '../../../../source/types';

  // Form state
  let modelName: string = '';
  let loading: boolean = false;
  let error: string | null = null;
  let models: Model[] = [];

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
      await goto(`/model/${model.id}/layers`);
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  // Load existing models from server
  async function loadModels(): Promise<void> {
    try {
      const res = await client.query({
        query: GET_MODELS,
        fetchPolicy: 'network-only'
      });
      models = res.data?.getModels || [];
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  }

  // Load models on component mount
  onMount(() => {
    loadModels();
  });
</script>

<div class="h-screen flex flex-col p-6">
  <!-- Header -->
  <div class="flex justify-between items-center mb-6">
    <!-- Left-aligned Icon + Title -->
    <div class="flex items-center space-x-3">
      <img src="/brain.png" alt="Brain Icon" class="w-8 h-8" />
      <h1 class="text-3xl font-bold">Synapse Works</h1>
    </div>

    <!-- Right-aligned Profile Icon -->
    <div class="w-10 h-10 rounded-full border border-gray-600 flex items-center justify-center">
      👤
    </div>
  </div>

  <!-- Main content -->
  <div class="flex flex-1 border rounded-lg overflow-hidden shadow-md">
    <!-- Left Panel: Create Model -->
    <div class="w-1/2 p-6 border-r">
      <h2 class="text-2xl font-semibold mb-4">Create Model</h2>

      <form on:submit|preventDefault={createModel} class="space-y-4">
        <input
          type="text"
          bind:value={modelName}
          placeholder="Enter the model name"
          required
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />

        <button 
          type="submit" 
          disabled={loading || !modelName.trim()}
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Creating...' : 'Create Model'}
        </button>

        {#if error}
          <div class="p-2 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        {/if}
      </form>
    </div>

    <!-- Right Panel: Existing Models -->
    <div class="w-1/2 p-6">
      <h2 class="text-xl font-semibold mb-4">Your Models</h2>

      {#if models.length === 0}
        <p class="text-gray-500">No models found. Create your first model to get started!</p>
      {:else}
        <div class="space-y-3">
          {#each models as model}
            <div class="p-4 bg-gray-50 rounded-lg border hover:bg-gray-100 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <h3 class="font-semibold text-lg">{model.name}</h3>
                  <p class="text-sm text-gray-600">ID: {model.id}</p>
                  <p class="text-sm text-gray-600">
                    Layers: {model.layers_config?.length || 0}
                  </p>
                  {#if model.dataset_config}
                    <p class="text-sm text-gray-600">
                      Dataset: {model.dataset_config.name}
                    </p>
                  {/if}
                </div>
                <div class="flex flex-col gap-2">
                  <!-- Navigate to layer configuration -->
                  <button
                    on:click={() => goto(`/model/${model.id}/layers`)}
                    class="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700"
                  >
                    View
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