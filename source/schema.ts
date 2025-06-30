// source/schema.ts
import { makeExecutableSchema } from '@graphql-tools/schema';
import { parse } from 'graphql';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { resolvers } from './resolvers.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const schemasDir = path.join(__dirname, 'schemas');

console.log(`[synapse][schema-loader]: Loading from: ${schemasDir}`);

// Get all .graphql files
const getAllGraphQLFiles = (dir: string): string[] => {
    const files: string[] = [];
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const item of items) {
        const fullPath = path.join(dir, item.name);
        if (item.isDirectory()) {
            files.push(...getAllGraphQLFiles(fullPath));
        } else if (item.name.endsWith('.graphql')) {
            files.push(fullPath);
        }
    }
    return files;
};

const graphqlFiles = getAllGraphQLFiles(schemasDir);
console.log(`[synapse]: Found ${graphqlFiles.length} GraphQL files:`);

const typeDefs: string[] = [];

// Process each file individually to catch the problematic one
for (const filePath of graphqlFiles) {
    console.log(`[synapse]: Processing ${filePath}`);
    
    try {
        const content = fs.readFileSync(filePath, 'utf-8');
        console.log(`[synapse]: File size: ${content.length} characters`);
        console.log(`[synapse]: First 100 chars: ${content.substring(0, 100)}`);
        
        // Try to parse this individual file
        try {
            parse(content);
            console.log(`[synapse]: ${filePath} parsed successfully`);
            typeDefs.push(content);
        } catch (parseError) {
            console.error(`[synapse]: Parse error in ${filePath}:`);
            console.error(parseError);
            console.error(`[synapse]: File content:\n${content}`);
            throw new Error(`Invalid GraphQL in file: ${filePath}`);
        }
        
    } catch (error) {
        console.error(`[synapse]: Error reading ${filePath}:`, error);
        throw error;
    }
}

console.log(`[synapse]: Successfully loaded ${typeDefs.length} schema files`);
console.log(`[synapse]: Resolvers keys: ${Object.keys(resolvers || {}).join(', ')}`);

// Create executable schema
export const schema = makeExecutableSchema({
    typeDefs,
    resolvers
});

console.log(`[synapse]: Schema created successfully`);