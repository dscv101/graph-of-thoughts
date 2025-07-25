# Roadmap

> Our plans for evolving Model Context Protocol

<Info>Last updated: **2025-03-27**</Info>

The Model Context Protocol is rapidly evolving. This page outlines our current thinking on key priorities and direction for approximately **the next six months**, though these may change significantly as the project develops. To see what's changed recently, check out the **[specification changelog](/specification/2025-06-18/changelog/)**.

<Note>
  The ideas presented here are not commitments—we may solve these challenges differently than described, or some may not materialize at all. This is also not an *exhaustive* list; we may incorporate work that isn't mentioned here.
</Note>

We value community participation! Each section links to relevant discussions where you can learn more and contribute your thoughts.

For a technical view of our standardization process, visit the [Standards Track](https://github.com/orgs/modelcontextprotocol/projects/2/views/2) on GitHub, which tracks how proposals progress toward inclusion in the official [MCP specification](https://spec.modelcontextprotocol.io).

## Validation

To foster a robust developer ecosystem, we plan to invest in:

* **Reference Client Implementations**: demonstrating protocol features with high-quality AI applications
* **Compliance Test Suites**: automated verification that clients, servers, and SDKs properly implement the specification

These tools will help developers confidently implement MCP while ensuring consistent behavior across the ecosystem.

## Registry

For MCP to reach its full potential, we need streamlined ways to distribute and discover MCP servers.

We plan to develop an [**MCP Registry**](https://github.com/orgs/modelcontextprotocol/discussions/159) that will enable centralized server discovery and metadata. This registry will primarily function as an API layer that third-party marketplaces and discovery services can build upon.

## Agents

As MCP increasingly becomes part of agentic workflows, we're exploring [improvements](https://github.com/modelcontextprotocol/specification/discussions/111) such as:

* **[Agent Graphs](https://github.com/modelcontextprotocol/specification/discussions/94)**: enabling complex agent topologies through namespacing and graph-aware communication patterns
* **Interactive Workflows**: improving human-in-the-loop experiences with granular permissioning, standardized interaction patterns, and [ways to directly communicate](https://github.com/modelcontextprotocol/specification/issues/97) with the end user

## Multimodality

Supporting the full spectrum of AI capabilities in MCP, including:

* **Additional Modalities**: video and other media types
* **[Streaming](https://github.com/modelcontextprotocol/specification/issues/117)**: multipart, chunked messages, and bidirectional communication for interactive experiences

## Governance

We're implementing governance structures that prioritize:

* **Community-Led Development**: fostering a collaborative ecosystem where community members and AI developers can all participate in MCP's evolution, ensuring it serves diverse applications and use cases
* **Transparent Standardization**: establishing clear processes for contributing to the specification, while exploring formal standardization via industry bodies

## Get Involved

We welcome your contributions to MCP's future! Join our [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions) to share ideas, provide feedback, or participate in the development process.
