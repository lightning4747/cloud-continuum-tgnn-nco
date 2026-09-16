### Executive Summary

Modern telecommunication architectures, specifically 5G and emerging 6G systems, rely on Network Function Virtualization (NFV) and cloud-native paradigms to deliver flexible, high-performance services. Within this context, a Service Function Chain (SFC) defines an abstract policy specifying the precise topological sequence of virtualized network functions through which data packets must travel between a client and an application server. Cloud-Native Network Functions (CNFs) serve as the containerized implementations of these individual functions, such as User Plane Functions (UPF) or Access and Mobility Management Functions (AMF).

---

### Analysis of Base Literature

The underlying mechanisms governing SFC and CNF operational frameworks can be separated into inter-domain control-plane orchestration and intra-domain resource optimization:

#### 1. Inter-Domain Federation Framework

Research on cross-operator orchestration addresses the challenge of maintaining network slices and SFC continuity when a mobile user transitions across administrative boundaries (e.g., roaming between different Mobile Network Operators). Utilizing GSMA Operator Platforms (OP), 3GPP Service Based Architectures (SBA), and O-RAN components, this framework standardizes the negotiation and instantiation of SFCs in visited networks. It introduces Service Function as a Service (SFaaS) and container lifecycle hooks to automate the deployment of CNFs across heterogeneous operator domains.

#### 2. Intra-Domain CNF Placement Optimization

Research on resource allocation focuses on the NP-hard problem of mapping the logical CNFs of an SFC onto physical or edge cloud infrastructure under strict multidimensional constraints. Utilizing Denoising Diffusion Probabilistic Models (DDPM) combined with Graph Neural Networks (GNNs), this approach generates optimal placement strategies that minimize end-to-end packet latency and operational overhead while strictly adhering to node-level CPU, RAM, and link bandwidth limits.

---

### Architectural Impact of Dynamic Application State

While current base architectures optimize for static resource allocation and cross-domain control negotiation, the introduction of dynamic application state—comprising active session contexts, in-flight data buffers, TCP socket states, and ephemeral memory—presents critical operational bottlenecks.

* **State Disruption during Mobility:** Rerouting an SFC across operator boundaries via federation protocols induces session breakage if application state is not migrated synchronously with control-plane handoffs.


* **Placement Optimization Penalties:** Mathematical models using GNN/DDPM formulations for placement optimization become incomplete without accounting for state transfer costs. Moving a containerized CNF to a lower-latency compute node creates transient latency spikes and bandwidth saturation if the associated state payload is large.


* **Synchronization Overhead:** In-band state replication across untrusted or bandwidth-constrained edge links can degrade network throughput, undermining the latency gains achieved by algorithmic CNF placement.



---

### System Integration Matrix

| Architectural Layer | Base Mechanism | Dynamic State Challenge | Required Framework Extension |
| --- | --- | --- | --- |
| **Inter-Domain Federation** | GSMA OP & SFaaS Lifecycle Hooks

 | State loss during cross-MNO handoffs

 | Stateful export/import container hook extensions |
| **Intra-Domain Placement** | DDPM + GNN Optimization

 | Unaccounted state migration latency

 | State migration cost penalties in objective functions |
| **Data Plane Execution** | Ordered SFC Packet Routing

 | Packet drops during CNF state sync

 | Decoupled state stores or in-band state snapshotting |