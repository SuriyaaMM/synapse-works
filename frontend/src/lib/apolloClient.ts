import { ApolloClient, InMemoryCache } from '@apollo/client/core';

const client = new ApolloClient({
  uri: 'https://synapse-works-server.onrender.com/',
  cache: new InMemoryCache(),
});

export default client;
