<script lang="ts">
  import { goto } from '$app/navigation';
  import client from '$lib/apolloClient';
  import { CREATE_MODEL} from '$lib/mutations';
  import { GET_MODELS} from '$lib/queries';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';

  let modelName = '';
  let loading = false;
  let error: string | null = null;
  let models = [];

  async function createModel() {
    if (!modelName.trim()) {
      error = 'Please enter a model name';
      return;
    }

    loading = true;
    error = null;

    try {
      const res = await client.mutate({
        mutation: CREATE_MODEL,
        variables: { name: modelName.trim() }
      });

      const model = res.data?.createModel;
      if (!model?.id) {
        throw new Error('Model creation failed - no ID returned');
      }

      await goto(`/append-layer?modelId=${model.id}`);
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  async function loadModels() {
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

  onMount(() => {
    loadModels();
  });
</script>

<div class="h-screen flex flex-col p-6">
  <!-- Header -->
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-3xl font-bold">Welcome to Synapse Works</h1>
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
          placeholder="enter the model name"
          required
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />

        <button 
          type="submit" 
          disabled={loading || !modelName.trim()}
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Creating...' : 'Create'}
        </button>

        {#if error}
          <div class="p-2 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        {/if}
      </form>
    </div>

    <!-- Right Panel: Display Models -->
    <div class="w-1/2 p-6">
      <h2 class="text-xl font-semibold mb-4">Your Models</h2>

      {#if models.length === 0}
        <p class="text-gray-500">No models found.</p>
      {:else}
        <ul class="space-y-2">
          {#each models as model}
            <li class="p-2 bg-gray-100 rounded border">{model.name}</li>
          {/each}
        </ul>
      {/if}
    </div>
  </div>
</div>
