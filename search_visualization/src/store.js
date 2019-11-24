import Vue from 'vue';
import Vuex from 'vuex';
import { createSharedMutations } from 'vuex-electron';
import { DataSet } from 'vis-network';

Vue.use(Vuex);

export default new Vuex.Store({
  plugins: [
    createSharedMutations(),
  ],
  state: {
    nodes: new DataSet(),
    edges: new DataSet(),
  },
  getters: {
    nodes: state => state.nodes,
    edges: state => state.edges,
  },
  mutations: {
    addEvent: (state, event) => {
      if (event.event_type === 'NEW_NODE') {
        state.nodes.add({
          id: event.id,
          label: event.id,
        });
        if (event.predecessor) {
          state.edges.add({
            from: event.predecessor,
            to: event.id,
          });
        }
      }
    },
  },
  actions: {
    ADD_EVENT_ACTION: ({ commit }, payload) => {
      commit('addEvent', payload.event);
    },
  },
});
