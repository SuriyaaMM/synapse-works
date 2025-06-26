<script lang="ts">
  import client from '$lib/apolloClient';
  import { SAVE_MODEL } from '$lib/mutations';

  import './save-model.css';

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

<div class="container">
  <h1 class="title">Model Management</h1>
  
  <div class="space-y-6">
    <!-- Save Button -->
    <div class="button-group">
      <button
        on:click={saveModel}
        disabled={loading}
        class="button"
      >
        {#if loading}
          <div class="spinner"></div>
          Saving...
        {:else}
          💾 Save Model
        {/if}
      </button>
    </div>

    <!-- Success Message -->
    {#if successMessage}
      <div class="success-message">
        <span>✅</span>
        {successMessage}
      </div>
    {/if}

    <!-- Error Message -->
    {#if error}
      <div class="error-message">
        <span>❌</span>
        {error}
      </div>
    {/if}

    <!-- Help Section -->
    <div class="help-section">
      <h3 class="help-title">
        <span>💡</span>
        Help
      </h3>
      <div class="help-content">
        <p>
          <strong>Save Model:</strong> Saves the current model configuration to storage as a .json file.
          This preserves your model's structure and settings for future use.
        </p>
      </div>
    </div>
  </div>
</div>