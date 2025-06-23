import { Model } from "./types";
import { writeFile, readFile, mkdir } from "fs/promises";
import path  from "path";
import { enqueueMessage } from "./redisClient.js";

export async function serialize(model: Model, dirname: string = "./savefile") {

    if(!model) throw new Error(`[synapse][serialization]: Model was empty! nothing to serialize`);

    const jsonified_model = JSON.stringify(model, null, 4);
    const file_path = path.join(dirname, "savefile.json");

    await mkdir(dirname, { recursive: true });

    await writeFile(file_path, jsonified_model, 'utf-8');
    console.log(`[synapse][serialization]: Serialized to ${file_path}`);

    return true;
}

export async function deserialize(dirname: string = "./savefile"){

    const file_path = path.join(dirname, "savefile.json");

    const models_buffer = await readFile(file_path, 'utf-8');
    const parsed = JSON.parse(models_buffer);
    // Ensures model is always an array
    const models = Array.isArray(parsed) ? parsed : [parsed];
    console.log(`[synapse][serialization]: De-Serialized from ${file_path}`);
    return models;
}

export async function saveResolver(models: Model[], dirname: string = "./savefile"){
    for(const model of models){
        serialize(model, dirname);
    }

    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    const message = {
        event_type: "SERIALIZE_MODEL",
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
}

export async function loadResolver(dirname: string = "./savefile"){
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    const message = {
        event_type: "DESERIALIZE_MODEL",
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    return deserialize(dirname);
}