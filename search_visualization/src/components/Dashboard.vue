<template>
  <div class="dashboard">
    <div id="graph-view"></div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import { Network } from 'vis-network';

export default {
  name: 'Dashboard',
  components: {},
  data() {
    return {};
  },
  computed: {
    ...mapGetters([
      'nodes',
      'edges',
    ]),
  },
  methods: {
  },
  mounted() {
    const network = new Network(
      document.getElementById('graph-view'),
      { nodes: this.nodes, edges: this.edges },
      {},
    );
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.dashboard {
  width: 100%;
  height: 100%;
}
#graph-view {
  height: 100%;
  width: 100%;
}
</style>
