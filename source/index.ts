import { ApolloServer } from "@apollo/server";
import { startStandaloneServer } from "@apollo/server/standalone";
import { typeDefs } from "./schema.js";
import { resolvers } from "./resolvers.js";
import { connectRedis, dequeueMessage } from "./redisClient.js";


connectRedis();

const server = new ApolloServer({
    typeDefs,
    resolvers
})

const {url} = await startStandaloneServer(
    server, {
        listen: {port : 4000}
    }
)

console.log("Started server http://localhost:4000")