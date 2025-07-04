<script lang="ts">
  import client from '$lib/apolloClient';
  import { SET_DATASET_CONFIG } from '$lib/mutations';
  import type { Model } from '../../../../../../source/types/modelTypes';
  import { fetchModelDetails, modelDetails } from "../modelDetails";
  import { DATASET_OPTIONS, TRANSFORM_OPTIONS, CELEBA_TARGET_OPTIONS, VOC_IMAGE_SET_OPTIONS, VOC_YEAR_OPTIONS, DEFAULT_FORM_DATA, type DatasetFormData, type CSVPreviewData } from './dataset-config-types';
  import { validateDatasetForm, parseTransformInput, addTransformToInput, toggleArrayItem, updateTestSplit, createDatasetConfigInput, loadCSVPreview } from './dataset-utils';
  import { onMount } from 'svelte';
  import './dataset-config.css';

  let loading = false;
  let error: string | null = null;
  let result: Model | null = null;
  let formData: DatasetFormData = { ...DEFAULT_FORM_DATA };
  let csvPreview: CSVPreviewData = { data: [], columns: [], showingPreview: false, previewError: null, file: null };

  // Reactive statements
  $: formData.testSplit = updateTestSplit(formData.trainSplit);
  $: formData.mnistTransform = parseTransformInput(formData.mnistTransformInput);
  $: formData.cifar10Transform = parseTransformInput(formData.cifar10TransformInput);
  $: formData.imageFolderTransform = parseTransformInput(formData.imageFolderTransformInput);
  $: formData.csvFeatureColumns = parseTransformInput(formData.csvFeatureColumnsInput);
  $: formData.csvLabelColumns = parseTransformInput(formData.csvLabelColumnsInput);
  $: formData.celebaTargetType = parseTransformInput(formData.celebaTargetTypeInput);
  $: formData.celebaTransform = parseTransformInput(formData.celebaTransformInput);
  $: formData.celebaTargetTransform = parseTransformInput(formData.celebaTargetTransformInput);
  $: formData.vocsegmentationTransform = parseTransformInput(formData.vocsegmentationTransformInput);
  $: formData.vocsegmentationTargetTransform = parseTransformInput(formData.vocsegmentationTargetTransformInput);

  onMount(() => {
    fetchModelDetails();
  });

  async function setDatasetConfig() {
    const validationError = validateDatasetForm(formData);
    if (validationError) { error = validationError; return; }
    
    loading = true; error = null; result = null;
    try {
      const res = await client.mutate({
        mutation: SET_DATASET_CONFIG,
        variables: { datasetConfig: createDatasetConfigInput(formData) }
      });
      if (!res.data?.setDataset) throw new Error('Failed to set dataset configuration');
      result = res.data.setDataset;
      await fetchModelDetails();
    } catch (err: any) {
      error = err.message || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  function addTransform(type: 'mnist' | 'cifar10' | 'celeba' | 'vocsegmentation' | 'image_folder', transform: string, targetTransform: boolean = false) {
    const suffix = targetTransform ? 'TargetTransformInput' : 'TransformInput';
    const currentSuffix = targetTransform ? 'TargetTransform' : 'Transform';
    const key = `${type}${suffix}` as keyof DatasetFormData;
    const currentKey = `${type}${currentSuffix}` as keyof DatasetFormData;
    formData[key] = addTransformToInput(formData[key] as string, formData[currentKey] as string[], transform);
  }

  function addCelebaTargetType(targetType: string) {
    formData.celebaTargetType = toggleArrayItem(formData.celebaTargetType, targetType);
    formData.celebaTargetTypeInput = formData.celebaTargetType.join(', ');
  }

  async function handleCSVPreview() {
    if (!csvPreview.file) return;
    csvPreview.showingPreview = true; csvPreview.previewError = null;
    try {
      const preview = await loadCSVPreview(csvPreview.file);
      csvPreview = { ...csvPreview, ...preview };
    } catch (err: any) {
      csvPreview.previewError = err.message; csvPreview.data = []; csvPreview.columns = [];
    } finally {
      csvPreview.showingPreview = false;
    }
  }

  function toggleColumn(column: string, type: 'feature' | 'label') {
    if (type === 'feature') {
      formData.csvFeatureColumns = toggleArrayItem(formData.csvFeatureColumns, column);
      formData.csvFeatureColumnsInput = formData.csvFeatureColumns.join(', ');
    } else {
      formData.csvLabelColumns = toggleArrayItem(formData.csvLabelColumns, column);
      formData.csvLabelColumnsInput = formData.csvLabelColumns.join(', ');
    }
  }
</script>

<div class="dataset-config-container">
  <h1 class="dataset-config-heading">Dataset Configuration</h1>

  <div class="space-y-6">
    {#if $modelDetails}
      <div class="dataset-config-overview">
        <h3 class="font-semibold mb-2">Model Overview</h3>
        <p class="dataset-config-info">Model Name: <span>{$modelDetails.name}</span></p>
        {#if $modelDetails.module_graph}
          <p class="dataset-config-info">
            Layers: <span>{$modelDetails.module_graph?.layers?.length || 0}</span>
            Connections: <span>{$modelDetails.module_graph?.edges?.length || 0}</span>
          </p>
        {:else}
          <p class="dataset-config-warning">⚠️ No module graph found.</p>
        {/if}
        {#if $modelDetails.train_config}
          <p class="dataset-config-info">Training: <span>{$modelDetails.train_config.epochs} Epochs, {$modelDetails.train_config.optimizer} optimizer</span></p>
        {:else}
          <p class="dataset-config-warning">⚠️ No training configuration found.</p>
        {/if}
      </div>
    {/if}

    <form on:submit|preventDefault={setDatasetConfig} class="dataset-config-form space-y-6">
      <div>
        <label for="datasetName" class="dataset-config-label">Dataset <span class="text-red-500">*</span></label>
        <select id="datasetName" bind:value={formData.datasetName} required class="dataset-config-select" disabled={loading}>
          {#each DATASET_OPTIONS as option}<option value={option.value}>{option.label}</option>{/each}
        </select>
      </div>

      <div class="dataset-config-grid">
        <div>
          <label for="batchSize" class="dataset-config-label">Batch Size <span class="text-red-500">*</span></label>
          <input id="batchSize" type="number" bind:value={formData.batchSize} required min="1" max="1024" class="dataset-config-input" disabled={loading} />
        </div>
        <div>
          <label for="trainSplit" class="dataset-config-label">Train Split <span class="text-red-500">*</span></label>
          <input id="trainSplit" type="number" bind:value={formData.trainSplit} step="0.1" required min="0.1" max="0.9" class="dataset-config-input" disabled={loading} />
        </div>
        <div>
          <label for="testSplit" class="dataset-config-label">Test Split <span class="text-red-500">*</span></label>
          <input id="testSplit" type="number" bind:value={formData.testSplit} class="dataset-config-input" disabled readonly />
        </div>
      </div>

      <div class="dataset-config-checkbox">
        <input id="shuffle" type="checkbox" bind:checked={formData.shuffle} class="h-4 w-4" disabled={loading} />
        <label for="shuffle" class="ml-2 block text-sm">Shuffle dataset</label>
      </div>

      {#if formData.datasetName === 'mnist'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">MNIST Dataset Settings</h3>
          <div class="space-y-4">
            <div>
              <label for="mnistRoot" class="dataset-config-label">Data Root Path <span class="text-red-500">*</span></label>
              <input id="mnistRoot" type="text" bind:value={formData.mnistRoot} required class="dataset-config-input" disabled={loading} />
            </div>
            <div>
              <label for="mnistTransformInput" class="dataset-config-label">Transforms</label>
              <input id="mnistTransformInput" type="text" bind:value={formData.mnistTransformInput} placeholder="ToTensor, Normalize" class="dataset-config-input" disabled={loading} />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.mnistTransform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('mnist', transform)} disabled={loading || formData.mnistTransform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div class="dataset-config-grid">
              <div class="dataset-config-checkbox">
                <input id="mnistTrain" type="checkbox" bind:checked={formData.mnistTrain} class="h-4 w-4" disabled={loading} />
                <label for="mnistTrain" class="ml-2 text-sm">Use training set</label>
              </div>
              <div class="dataset-config-checkbox">
                <input id="mnistDownload" type="checkbox" bind:checked={formData.mnistDownload} class="h-4 w-4" disabled={loading} />
                <label for="mnistDownload" class="ml-2 text-sm">Auto-download</label>
              </div>
            </div>
          </div>
        </div>
      {/if}

      {#if formData.datasetName === 'celeba'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">CelebA Dataset Settings</h3>
          <div class="space-y-4">
            <div>
              <label for="celebaRoot" class="dataset-config-label">Data Root Path <span class="text-red-500">*</span></label>
              <input id="celebaRoot" type="text" bind:value={formData.celebaRoot} required class="dataset-config-input" disabled={loading} />
            </div>
            <div>
              <label for="celebaTargetTypeInput" class="dataset-config-label">Target Types (Optional)</label>
              <input id="celebaTargetTypeInput" type="text" bind:value={formData.celebaTargetTypeInput} class="dataset-config-input" disabled={loading} placeholder="attr, identity, bbox, landmarks" />
              <div class="mt-2 flex gap-2">
                {#each CELEBA_TARGET_OPTIONS as targetType}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.celebaTargetType.includes(targetType) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addCelebaTargetType(targetType)} disabled={loading}>
                    {targetType}
                  </button>
                {/each}
              </div>
            </div>
            <div>
              <label for="celebaTransformInput" class="dataset-config-label">Transforms (Optional)</label>
              <input id="celebaTransformInput" type="text" bind:value={formData.celebaTransformInput} class="dataset-config-input" disabled={loading} placeholder="ToTensor, Normalize" />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.celebaTransform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('celeba', transform)} disabled={loading || formData.celebaTransform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div>
              <label for="celebaTargetTransformInput" class="dataset-config-label">Target Transforms (Optional)</label>
              <input id="celebaTargetTransformInput" type="text" bind:value={formData.celebaTargetTransformInput} class="dataset-config-input" disabled={loading} placeholder="ToTensor, Normalize" />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.celebaTargetTransform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('celeba', transform, true)} disabled={loading || formData.celebaTargetTransform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div class="dataset-config-checkbox">
              <input id="celebaDownload" type="checkbox" bind:checked={formData.celebaDownload} class="h-4 w-4" disabled={loading} />
              <label for="celebaDownload" class="ml-2 text-sm">Auto-download</label>
            </div>
          </div>
        </div>
      {/if}

      {#if formData.datasetName === 'vocsegmentation'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">VOC Segmentation Dataset Settings</h3>
          <div class="space-y-4">
            <div>
              <label for="vocsegmentationRoot" class="dataset-config-label">Data Root Path <span class="text-red-500">*</span></label>
              <input id="vocsegmentationRoot" type="text" bind:value={formData.vocsegmentationRoot} required class="dataset-config-input" disabled={loading} />
            </div>
            <div class="dataset-config-grid">
              <div>
                <label for="vocsegmentationImageSet" class="dataset-config-label">Image Set (Optional)</label>
                <select id="vocsegmentationImageSet" bind:value={formData.vocsegmentationImageSet} class="dataset-config-select" disabled={loading}>
                  <option value="">Select image set...</option>
                  {#each VOC_IMAGE_SET_OPTIONS as imageSet}
                    <option value={imageSet}>{imageSet}</option>
                  {/each}
                </select>
              </div>
              <div>
                <label for="vocsegmentationYear" class="dataset-config-label">Year (Optional)</label>
                <select id="vocsegmentationYear" bind:value={formData.vocsegmentationYear} class="dataset-config-select" disabled={loading}>
                  <option value="">Select year...</option>
                  {#each VOC_YEAR_OPTIONS as year}
                    <option value={year}>{year}</option>
                  {/each}
                </select>
              </div>
            </div>
            <div>
              <label for="vocsegmentationTransformInput" class="dataset-config-label">Transforms (Optional)</label>
              <input id="vocsegmentationTransformInput" type="text" bind:value={formData.vocsegmentationTransformInput} class="dataset-config-input" disabled={loading} placeholder="ToTensor, Normalize" />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.vocsegmentationTransform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('vocsegmentation', transform)} disabled={loading || formData.vocsegmentationTransform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div>
              <label for="vocsegmentationTargetTransformInput" class="dataset-config-label">Target Transforms (Optional)</label>
              <input id="vocsegmentationTargetTransformInput" type="text" bind:value={formData.vocsegmentationTargetTransformInput} class="dataset-config-input" disabled={loading} placeholder="ToTensor, Normalize" />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.vocsegmentationTargetTransform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('vocsegmentation', transform, true)} disabled={loading || formData.vocsegmentationTargetTransform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div class="dataset-config-checkbox">
              <input id="vocsegmentationDownload" type="checkbox" bind:checked={formData.vocsegmentationDownload} class="h-4 w-4" disabled={loading} />
              <label for="vocsegmentationDownload" class="ml-2 text-sm">Auto-download</label>
            </div>
          </div>
        </div>
      {/if}

      {#if formData.datasetName === 'cifar10'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">CIFAR-10 Dataset Settings</h3>
          <div class="space-y-4">
            <div>
              <label for="cifar10Root" class="dataset-config-label">Data Root Path <span class="text-red-500">*</span></label>
              <input id="cifar10Root" type="text" bind:value={formData.cifar10Root} required class="dataset-config-input" disabled={loading} />
            </div>
            <div>
              <label for="cifar10TransformInput" class="dataset-config-label">Transforms</label>
              <input id="cifar10TransformInput" type="text" bind:value={formData.cifar10TransformInput} class="dataset-config-input" disabled={loading} />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.cifar10Transform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('cifar10', transform)} disabled={loading || formData.cifar10Transform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div class="dataset-config-grid">
              <div class="dataset-config-checkbox">
                <input id="cifar10Train" type="checkbox" bind:checked={formData.cifar10Train} class="h-4 w-4" disabled={loading} />
                <label for="cifar10Train" class="ml-2 text-sm">Use training set</label>
              </div>
              <div class="dataset-config-checkbox">
                <input id="cifar10Download" type="checkbox" bind:checked={formData.cifar10Download} class="h-4 w-4" disabled={loading} />
                <label for="cifar10Download" class="ml-2 text-sm">Auto-download</label>
              </div>
            </div>
          </div>
        </div>
      {/if}

      {#if formData.datasetName === 'custom_csv'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">Custom CSV Dataset Settings</h3>
          <div class="space-y-4">
            <div class="flex space-x-2">
              <input type="file" accept=".csv" on:change={(e) => {
                const target = e.target as HTMLInputElement;
                csvPreview.file = target?.files?.[0] || null;
                if (csvPreview.file) formData.csvRoot = csvPreview.file.name;
              }} class="dataset-config-input" disabled={loading} />
              <button type="button" on:click={handleCSVPreview} disabled={loading || !csvPreview.file} class="px-4 py-2 bg-blue-500 text-white rounded">
                {csvPreview.showingPreview ? 'Loading...' : 'Preview'}
              </button>
            </div>

            {#if csvPreview.previewError}
              <div class="p-3 bg-red-100 text-red-700 rounded">Error: {csvPreview.previewError}</div>
            {/if}

            {#if csvPreview.data.length > 0}
              <div class="border rounded overflow-hidden">
                <div class="bg-gray-50 px-4 py-2"><h4 class="font-medium">CSV Preview</h4></div>
                <div class="overflow-x-auto max-h-64">
                  <table class="min-w-full text-sm">
                    <thead class="bg-gray-100">
                      <tr>
                        {#each csvPreview.columns as column}
                          <th class="px-3 py-2 text-left cursor-pointer hover:bg-gray-200 
                                    {formData.csvFeatureColumns.includes(column) ? 'bg-green-200' : ''}
                                    {formData.csvLabelColumns.includes(column) ? 'bg-orange-200' : ''}"
                              on:click={() => toggleColumn(column, 'feature')}
                              on:dblclick={() => toggleColumn(column, 'label')}>
                            {column}
                            {#if formData.csvFeatureColumns.includes(column)}<span class="text-xs">(F)</span>{/if}
                            {#if formData.csvLabelColumns.includes(column)}<span class="text-xs">(L)</span>{/if}
                          </th>
                        {/each}
                      </tr>
                    </thead>
                    <tbody>
                      {#each csvPreview.data as row, i}
                        <tr class="border-t {i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}">
                          {#each csvPreview.columns as column}
                            <td class="px-3 py-2">{row[column] || ''}</td>
                          {/each}
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              </div>
              <div class="dataset-config-grid">
                <div>
                  <label for="csvFeatureColumnsInput" class="dataset-config-label">Feature Columns <span class="text-red-500">*</span></label>
                  <input id="csvFeatureColumnsInput" type="text" bind:value={formData.csvFeatureColumnsInput} required class="dataset-config-input" disabled readonly />
                </div>
                <div>
                  <label for="csvLabelColumnsInput" class="dataset-config-label">Label Columns <span class="text-red-500">*</span></label>
                  <input id="csvLabelColumnsInput" type="text" bind:value={formData.csvLabelColumnsInput} required class="dataset-config-input" disabled readonly />
                </div>
              </div>
            {/if}

            <div class="dataset-config-checkbox">
              <input id="csvIsRegressionTask" type="checkbox" bind:checked={formData.csvIsRegressionTask} class="h-4 w-4" disabled={loading} />
              <label for="csvIsRegressionTask" class="ml-2 text-sm">Regression task</label>
            </div>
          </div>
        </div>
      {/if}

      {#if formData.datasetName === 'image_folder'}
        <div class="dataset-config-section">
          <h3 class="dataset-config-section-title">Image Folder Dataset Settings</h3>
          <div class="space-y-4">
            <div>
              <label for= "imageFolderRoot" class="dataset-config-label">Images Root Path <span class="text-red-500">*</span></label>
              <input id="imageFolderRoot" type="text" bind:value={formData.imageFolderRoot} required class="dataset-config-input" disabled={loading} />
            </div>
            <div>
              <label class="dataset-config-label">Transforms</label>
              <input type="text" bind:value={formData.imageFolderTransformInput} class="dataset-config-input" disabled={loading} />
              <div class="mt-2 flex gap-2">
                {#each TRANSFORM_OPTIONS as transform}
                  <button type="button" class="px-2 py-1 text-xs rounded {formData.imageFolderTransform.includes(transform) ? 'bg-blue-200' : 'bg-gray-200'}" 
                          on:click={() => addTransform('image_folder', transform)} disabled={loading || formData.imageFolderTransform.includes(transform)}>
                    {transform}
                  </button>
                {/each}
              </div>
            </div>
            <div class="dataset-config-checkbox">
              <input type="checkbox" bind:checked={formData.imageFolderAllowEmpty} class="h-4 w-4" disabled={loading} />
              <label class="ml-2 text-sm">Allow empty folders</label>
            </div>
          </div>
        </div>
      {/if}

      <div class="flex pt-6">
        <button type="submit" disabled={loading} class="dataset-config-button">
          {loading ? 'Saving...' : 'Save Dataset Configuration'}
        </button>
      </div>
    </form>

    {#if error}<div class="dataset-config-form-error">{error}</div>{/if}
    {#if result}<div class="dataset-config-success">Configuration Saved Successfully</div>{/if}
  </div>
</div>