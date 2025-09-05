# CompressARC Computational Graph

This document provides a visual representation of the computational flow in the CompressARC architecture.

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARC Task/Problem                           │
│  (Contains Multiple Input/Output Examples for a Single Puzzle)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Task Preprocessing (Task)                     │
│ - Extract problem shapes                                        │
│ - Create problem tensor                                         │
│ - Set up multitensor system                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ARCCompressor Initialization                   │
│ - Initialize weights                                            │
│ - Set up layer structures                                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Training Loop                           │
│ - For each iteration:                                           │
│   - Forward pass                                                │
│   - Compute loss                                                │
│   - Backward pass                                               │
│   - Update weights                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Forward Pass Flow

```
┌─────────────────────────────┐
│   Learned Latent Variables  │
│   (multiposteriors)         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Decode Latents Layer      │
│   - KL divergence computed  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Multi-Layer Processing    │
│   (Repeated n_layers times) │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Linear Heads              │
│   - Output logits           │
│   - x_mask, y_mask          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Compute Loss              │
│   - KL divergence           │
│   - Reconstruction error    │
└─────────────────────────────┘
```

## Multi-Layer Processing (Detail)

```
┌─────────────────────────────┐
│   Layer Input               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Share Up                  │
│   - Info from lower dims    │
│     to higher dims          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Softmax Layer             │
│   - Apply with pre-norm     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Directional Layers        │
│   - Cummax                  │
│   - Shift                   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Direction Share           │
│   - Share across directions │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Nonlinear Layer           │
│   - Add nonlinearity        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Share Down                │
│   - Info from higher dims   │
│     to lower dims           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Normalization             │
│   - Normalize outputs       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Layer Output              │
└─────────────────────────────┘
```

## MultiTensor System and Information Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      MultiTensor System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Dimension 0: Examples                                         │
│   Dimension 1: Colors                                           │
│   Dimension 2: Directions                                       │
│   Dimension 3: X coordinates                                    │
│   Dimension 4: Y coordinates                                    │
│                                                                 │
│   Valid dimension combinations form a hierarchy:                │
│                                                                 │
│   [1,1,1,1,1] ◄───┐                                            │
│        ▲           │                                            │
│        │           │                                            │
│   [1,1,1,1,0] ◄─┐  │                                           │
│        ▲        │  │                                            │
│        │        │  │                                            │
│   [1,1,1,0,1] ◄─┼──┘                                           │
│        ▲        │                                               │
│        │        │                                               │
│   [1,1,1,0,0] ◄─┘                                              │
│        ▲                                                        │
│        │                                                        │
│   [1,1,0,0,0]                                                   │
│        ▲                                                        │
│        │                                                        │
│   [1,0,0,0,0]                                                   │
│        ▲                                                        │
│        │                                                        │
│   [0,1,0,0,0]                                                   │
│                                                                 │
│   Share Up: Information flows upward through this hierarchy     │
│   Share Down: Information flows downward through this hierarchy │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Cross-Example Information Sharing

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Example Information Sharing                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Example-Specific Tensors (dims[0]=1)                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │Example 1│    │Example 2│    │Example 3│                    │
│   └────┬────┘    └────┬────┘    └────┬────┘                    │
│        │              │              │                          │
│        │              │              │                          │
│        ▼              ▼              ▼                          │
│   ┌─────────────────────────────────────────────┐              │
│   │           Share Down Operation              │              │
│   └─────────────────┬───────────────────────────┘              │
│                     │                                           │
│                     ▼                                           │
│   ┌─────────────────────────────────────────────┐              │
│   │       Example-Agnostic Tensor (dims[0]=0)    │              │
│   │      (Captures patterns across examples)     │              │
│   └─────────────────┬───────────────────────────┘              │
│                     │                                           │
│                     ▼                                           │
│   ┌─────────────────────────────────────────────┐              │
│   │            Share Up Operation               │              │
│   └─────────────────┬───────────────────────────┘              │
│                     │                                           │
│        ┌────────────┼────────────┐                             │
│        │            │            │                             │
│        ▼            ▼            ▼                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │Example 1│    │Example 2│    │Example 3│                    │
│   │(Updated)│    │(Updated)│    │(Updated)│                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Reconstruction Error Computation

```
┌─────────────────────────────────────────────────────────────────┐
│              Reconstruction Error Calculation                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each example & each in/out mode:                           │
│                                                                 │
│  ┌───────────────────┐     ┌───────────────────┐               │
│  │   Model Output    │     │   Target Grid     │               │
│  │  (logits, masks)  │     │  (problem_slice)  │               │
│  └─────────┬─────────┘     └─────────┬─────────┘               │
│            │                         │                          │
│            └─────────┬───────────────┘                          │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │  1. Determine if grid size is uncertain │                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  2. Compute position log probabilities  │                   │
│  │     using mask_select_logprobs          │                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  3. If grid size uncertain, compute     │                   │
│  │     probabilities for all possible sizes│                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  4. For each possible grid position:    │                   │
│  │     - Calculate positional log probs    │                   │
│  │     - Calculate color cross-entropy     │                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  5. Aggregate across all positions      │                   │
│  │     using logsumexp                     │                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  Reconstruction Error Component         │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Full Loss Function

```
┌─────────────────────────────────────────────────────────────────┐
│                     Loss Function                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐       ┌─────────────────┐                │
│   │     KL Loss     │       │ Reconstruction  │                │
│   │  (from latents) │       │     Error       │                │
│   └────────┬────────┘       └────────┬────────┘                │
│            │                         │                          │
│            │                         │                          │
│            ▼                         ▼                          │
│   ┌─────────────────┐       ┌─────────────────┐                │
│   │  Sum over all   │       │ Weighted by 10  │                │
│   │  KL components  │       │                 │                │
│   └────────┬────────┘       └────────┬────────┘                │
│            │                         │                          │
│            └─────────────┬───────────┘                          │
│                          │                                      │
│                          ▼                                      │
│                  ┌───────────────┐                             │
│                  │  Total Loss   │                             │
│                  └───────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Through Entire System

```
┌─────────────────────────────────────────────────────────────────┐
│                  Complete Data Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Task Examples (Inputs/Outputs)                                 │
│           │                                                     │
│           ▼                                                     │
│  Learned Latent Variables                                       │
│           │                                                     │
│           ▼                                                     │
│  Decode Latents & Compute KL                                    │
│           │                                                     │
│           ▼                                                     │
│  ┌───────────────────────┐                                     │
│  │ Multi-Layer Processing │                                     │
│  │                       │                                     │
│  │  ┌─────────────────┐  │                                     │
│  │  │    Share Up     │  │                                     │
│  │  └────────┬────────┘  │                                     │
│  │           │           │                                     │
│  │           ▼           │                                     │
│  │  ┌─────────────────┐  │                                     │
│  │  │Processing Layers│  │                                     │
│  │  └────────┬────────┘  │                                     │
│  │           │           │                                     │
│  │           ▼           │                                     │
│  │  ┌─────────────────┐  │                                     │
│  │  │   Share Down    │  │ ◄─────┐                             │
│  │  └────────┬────────┘  │       │                             │
│  └───────────┬───────────┘       │                             │
│              │                   │                             │
│              ▼                   │                             │
│         Repeat Layers ───────────┘                             │
│              │                                                 │
│              ▼                                                 │
│  Linear Heads for Output                                       │
│              │                                                 │
│              ▼                                                 │
│  Compute Reconstruction Error                                  │
│              │                                                 │
│              ▼                                                 │
│  Combine with KL for Total Loss                                │
│              │                                                 │
│              ▼                                                 │
│  Backpropagation & Weight Updates                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
