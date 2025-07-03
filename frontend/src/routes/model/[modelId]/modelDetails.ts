import client from "$lib/apolloClient";
import { GET_MODEL } from "$lib/queries";
import type { Model } from "../../../../../source/types/modelTypes";

export let error: string | null = null;
export let modelDetails: Model | null = null;

export async function fetchModelDetails() {
    try {
      const response = await client.query({
        query: GET_MODEL,
        fetchPolicy: 'network-only'
      });
      
      modelDetails = response.data?.getModel;
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }