import { writable } from 'svelte/store';
import client from "$lib/apolloClient";
import { GET_MODEL } from "$lib/queries";
import type { Model } from "../../../../../source/types/modelTypes";

export const error = writable<string | null>(null);
export const modelDetails = writable<Model | null>(null);

export async function fetchModelDetails() {
    try {
        const response = await client.query({
            query: GET_MODEL,
            fetchPolicy: 'network-only'
        });

        modelDetails.set(response.data?.getModel || null);
    } catch (err) {
        console.error('Error fetching model details:', err);
        error.set('Failed to fetch model details');
    }
}