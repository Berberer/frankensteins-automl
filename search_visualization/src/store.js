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
            id: event.predecessor + event.id,
            from: event.predecessor,
            to: event.id,
            value: 1,
          });
        }
      } else if (event.event_type === 'WEIGHT_UPDATE') {
        state.edges.update({
          id: event.predecessor + event.id,
          from: event.from,
          to: event.to,
          value: event.weight * 10,
        });
      }
    },
  },
  actions: {
    ADD_EVENT_ACTION: ({ commit }, payload) => {
      commit('addEvent', payload.event);
    },
  },
});
