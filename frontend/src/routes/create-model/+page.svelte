<script lang="ts">
  import { goto } from '$app/navigation';
  import client from '$lib/apolloClient';
  import { CREATE_MODEL } from '$lib/mutations';

  let modelName = '';
  let loading = false;
  let error: string | null = null;

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

      // Navigate to append-layer page with model id as query param
      await goto(`/append-layer?modelId=${model.id}`);
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }
</script>

<div class="container mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6">Create Model</h1>

  <form on:submit|preventDefault={createModel} class="space-y-4">
    <div>
      <input
        type="text"
        bind:value={modelName}
        placeholder="Enter model name"
        required
        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={loading}
      />
    </div>
    
    <button 
      type="submit" 
      disabled={loading || !modelName.trim()}
      class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {loading ? 'Creating...' : 'Create Model'}
    </button>
  </form>

  {#if error}
    <div class="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
      {error}
    </div>
  {/if}
</div>