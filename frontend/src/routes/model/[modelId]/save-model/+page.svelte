<script lang="ts">
  import client from '$lib/apolloClient';
  import { SAVE_MODEL } from '$lib/mutations';

  // State variables
  let loading = false;
  let error: string | null = null;
  let successMessage: string | null = null;

  
  // Saves the current model
  async function saveModel() {
    loading = true;
    error = null;
    successMessage = null;

    try {
      const response = await client.mutate({
        mutation: SAVE_MODEL
      });

      successMessage = 'Model saved successfully!';

    } catch (err: any) {
      error = err.message || 'Failed to save model';
    } finally {
      loading = false;
    }
  }
</script>

<div class="container mx-auto p-1">
  <h1 class="text-3xl font-bold mb-4">Model Management</h1>
  
  <div class="space-y-6">
    <!-- Save Button -->
    <div class="flex gap-4 flex-wrap">
      <button
        on:click={saveModel}
        disabled={loading}
        class="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {#if loading}
          <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          Saving...
        {:else}
          💾 Save Model
        {/if}
      </button>
    </div>

    <!-- Success Message -->
    {#if successMessage}
      <div class="p-4 bg-green-100 border border-green-400 text-green-700 rounded-md">
        <div class="flex items-center gap-2">
          <span class="text-green-600">✅</span>
          {successMessage}
        </div>
      </div>
    {/if}

    <!-- Error Message -->
    {#if error}
      <div class="p-4 bg-red-100 border border-red-400 text-red-700 rounded-md">
        <div class="flex items-center gap-2">
          <span class="text-red-600">❌</span>
          {error}
        </div>
      </div>
    {/if}

    <!-- Help Section -->
    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
      <h3 class="font-semibold text-gray-800 mb-2 flex items-center gap-2">
        <span>💡</span>
        Help
      </h3>
      <div class="text-sm text-gray-600 space-y-2">
        <p>
          <strong>Save Model:</strong> Saves the current model configuration to storage as .json file. 
          This preserves your model's structure and settings for future use.
        </p>
      </div>
    </div>
  </div>
</div>