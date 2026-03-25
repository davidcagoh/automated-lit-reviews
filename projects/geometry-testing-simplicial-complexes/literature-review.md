# Detecting Latent Geometry in Random Graphs and Simplicial Complexes: A Literature Review

## Overview

A central question in modern network science and high-dimensional statistics is whether an observed graph or simplicial complex reflects an underlying latent geometric structure, or whether its edges are merely the product of chance without spatial organisation. This review surveys the literature on this problem: hypothesis testing between geometric random graph models and non-geometric nulls, the statistical and computational thresholds governing detectability, spectral and subgraph-counting methods for exploiting geometry, recovery of latent positions from graph observations, and extensions to higher-dimensional simplicial complexes. The corpus spans foundational probabilistic machinery (1996–2013), the emergence of the geometry detection problem as a formal statistical question (2014–2018), and a wave of sharp threshold results and computational lower bounds (2019–2025).

---

## The Geometry Detection Problem: Models and Formulation

The canonical model for a graph with hidden geometry is the random geometric graph (RGG). In the spherical model $\mathsf{RGG}(n, \mathbb{S}^{d-1}, p)$, each of $n$ vertices is assigned an i.i.d. uniform position on the $(d-1)$-dimensional unit sphere, and two vertices are connected whenever their inner product exceeds a threshold $\tau^p_d$ chosen so that the expected edge density is $p$ [1]. The Erdős–Rényi model $G(n,p)$ serves as the non-geometric null: edges are drawn independently with probability $p$, with no spatial structure whatsoever. The geometry detection problem asks: given a single graph sampled from one of these two distributions, can we reliably distinguish them?

The foundational paper in this direction is [2], who formalised the hypothesis test $H_0: G \sim G(n,p)$ versus $H_1: G \sim G(n,p,d)$, and established two complementary results. On the upper side, they proposed a test based on *signed triangles* — a centred count of triangles — that achieves power in the dense regime ($p$ constant) when $d = o(n^3)$. On the lower side, they proved a detection lower bound by establishing a new bound on the total variation distance between a Wishart matrix $X^\top X$ and an appropriately normalised GOE matrix: when $d \gg n^3$, the two models are indistinguishable by any test. The paper also conjectured the optimal detection boundary in the sparse regime.

The relationship between RGGs and Erdős–Rényi in high dimensions was studied independently by [3], who showed that when $d \gtrsim \log^3 n$, the clique number of a high-dimensional RGG is very close to that of the corresponding ER graph, quantifying the sense in which high-dimensional random geometric graphs are approximately exchangeable. This was later substantially refined. [4] studied RGGs on spheres in the sparse regime, connecting the problem to the spherical Wishart matrix and showing that when $d$ grows slightly faster than $n$, the graph is comparable to ER. [1] proved a sharp sandwiching result: $\mathsf{RGG}(n, \mathbb{S}^{d-1}, p)$ can be sandwiched between two ER graphs in a strong coupling sense, with applications to sharp thresholds of monotone properties, robust testing, and enumeration problems.

The anisotropic setting was studied by [6], who considered vertices drawn from a high-dimensional Gaussian distribution rather than a sphere. They proved that detection is impossible when $d \geq \log n$, affirmatively resolving a conjecture of [7], who had earlier shown detection is possible when $d < \log n$. The threshold $d \sim \log n$ is thus sharp for anisotropic geometry detection. A related model, the *soft* or *noisy* RGG $G(n,p,d,q)$, where edges are added and deleted with noise probability $q$ on top of a geometric base, was analysed by [8], who characterised the phase transition for detecting latent structure as a function of $d$, $n$, $p$, and $q$.

For a broader class of *latent space graphs*, where the edge probability between vertices $i$ and $j$ is a general monotone increasing function of $\langle u_i, u_j \rangle$, [9] identified phase transitions for geometry detection using information-theoretic inequalities and concentration of measure. When the dimension is high or the variance parameter is large, the graph is statistically indistinguishable from Erdős–Rényi; in the low-dimensional or low-variance regime, a signed triangle statistic efficiently distinguishes the two. [10] approached the convergence question through entropy, proving via a multivariate CLT that RGGs on the torus converge to the ER ensemble in distribution as $d \to \infty$, and deriving the Shannon entropy of RGGs as a function of $n$ and $d$.

Random algebraic graphs [11, 12] extend the spherical model to general compact groups $G$ with connection function $\sigma: G \to [0,1]$. The convergence of these graphs to Erdős–Rényi is characterised by the Fourier coefficients of $\sigma$ on the group, providing a unified algebraic framework that contains the spherical RGG as a special case.

---

## Subgraph Counting as a Test for Latent Geometry

Subgraph statistics — counts of triangles, paths, cliques — are the most natural observables for detecting geometric structure, since geometry creates local clustering that distorts the counts relative to the ER null.

The signed triangle test of [2] was the first computationally efficient procedure for geometry detection. Its correctness relies on the fact that under RGG, the signed triangle count $\sum_{i<j<k} (A_{ij} - p)(A_{jk} - p)(A_{ik} - p)$ has variance of order $n^3 p^3 / d^{3/2}$, which dominates the ER baseline when $d = o(n^3)$.

[13] showed that triangle counts and clustering coefficients, despite their prevalence in network science, are *insufficient* to detect hyperbolic geometry in networks: two models with the same triangle count can differ strongly in their underlying geometry. They introduced *weighted triangles* — a statistic that weights each triangle by the reciprocal of the degrees of its vertices — and demonstrated both analytically and on real-world data that this statistic powerfully detects hyperbolic latent geometry where unweighted triangles fail.

[14] applied graph-based testing to the concrete problem of botnet detection. The null hypothesis is an RGG on a $d$-dimensional torus (representing a normal network); the alternative is a botnet in which a small number of vertices bypass geometric structure and connect randomly to every other vertex. They proposed two asymptotically optimal tests: an *isolated star test*, based on counting vertices of anomalously high degree, and an *average graph distance test*. The isolated star test significantly outperforms the distance test on moderate-sized networks and yields a robust identification scheme for individual botnet vertices.

[15] extended the detection problem to *localised* geometry: a power-law inhomogeneous random graph in which only a hidden subset of $k = o(n)$ vertices obey geometric connection rules. A test based on counting normalised triangles restricted to high-degree vertices correctly detects the geometric community and identifies its members with high probability. This is a harder regime than global geometry detection, since the geometric signal is concentrated in a vanishing fraction of the graph.

The behaviour of general subgraph counts over Poisson point processes was analysed by [16], who established a functional central limit theorem for subgraph counting processes, identifying the precise scaling under which normalised counts converge to Brownian motion. The upper tail of subgraph count distributions — critical for calibrating tests — was surveyed by [17], who compared seven techniques and derived essentially optimal bounds for the upper tails of $K_4$ and $C_4$ counts in $G(n,p)$.

Sharp thresholds for monotone properties of RGGs were characterised by [18]. [19] developed a unified class of goodness-of-fit tests against heterogeneous alternatives to the homogeneous ER model, deriving limiting distributions for a broad class of graph functionals and proving bootstrap consistency; their class subsumes several existing tests as special cases.

[20] studied the related problem of testing whether a correlation matrix is the identity, focusing on sparse alternatives. They derived general lower bounds and constructed computationally efficient near-optimal tests, with a direct connection to the clique number of high-dimensional RGGs: new lower bounds for this clique number arise as a by-product of the analysis.

---

## Spectral Methods and Fourier Analysis

Spectral statistics — eigenvalues and eigenvectors of adjacency and Laplacian matrices — offer a complementary route to geometry detection, particularly sensitive to global structure rather than local subgraph patterns.

The spectral properties of RGG adjacency matrices were first studied systematically by [21], who showed that RGG spectra differ strikingly from those of ER and scale-free graphs: the spectrum lacks symmetry about zero, and a striking singularity appears near $\pm 1.1$ in the thermodynamic limit.

The spectral gap of the higher-order Laplacian on the Linial–Meshulam random simplicial complex was analysed by [22], who proved that the gap between the smallest eigenvalues and the bulk is of constant order with high probability under $p \gg n^{-1/(d+1)}$, and that the global eigenvalue distribution converges to the semicircle law. The proof uses a Füredi–Komlós-type argument adapted to random simplicial complexes viewed as sparse random matrix models with dependent entries.

[5] established a comprehensive picture of the Fourier structure of RGGs: they computed the low-degree Fourier coefficients of $\mathsf{RGG}(n, \mathbb{S}^{d-1}, p)$ relative to ER, settled the low-degree polynomial complexity of distinguishing RGGs from ER (degree-$D$ polynomials require $d \lesssim n^{3/(D+1)}$), exhibited a statistical-computational gap for distinguishing RGG from a planted colouring model, and reproved spectral bounds as corollaries.

[23] proposed *higher-order spectral clustering* for geometric graphs, where standard spectral clustering is ineffective. Their algorithm uses the eigenvector corresponding to a higher-order eigenvalue of the adjacency matrix, and they proved weak and strong consistency under the Soft Geometric Block Model.

[24] proved that spectral embedding consistently estimates latent positions in random dot product graphs, and that subsequent $k$-nearest-neighbour classification achieves Bayes optimal error. [25] extended this to universal consistency for latent position graphs with universal kernel link functions. [26] analysed the Gaussian mixture block model and characterised when spectral clustering can recover the latent mixture structure.

The Lovász theta function was highlighted by [27] as a powerful unifying tool: the Lovász embedding places graph vertices on a spherical cap in a way that reflects graph structure, encoding spectral and combinatorial properties simultaneously.

---

## Detection Thresholds and the Role of Dimension

A unifying theme is that the dimension $d$ acts as a fundamental signal-to-noise parameter. For the dense spherical RGG, the signed triangle test achieves power when $d = o(n^3)$ and detection is impossible when $d \gg n^3$ [2]. For the anisotropic Gaussian model, the sharp threshold is at $d = \Theta(\log n)$ [6]. For the noisy model $G(n,p,d,q)$, [8] characterised the phase transition as a function of all four parameters.

[28] studied planted dense cycles, characterising information-theoretic thresholds for detection and recovery and establishing a provable statistical-computational gap. [29] proved the impossibility of inner product recovery when $d \approx nh(p)$, via a lower bound on the rate-distortion function of the Wishart distribution.

For networks with power-law degree distributions, [30, 31] studied geometric inhomogeneous random graphs (GIRGs). [13, 32] characterised phase transitions governing clique behaviour as functions of clique size $k$ and degree exponent $\tau$, showing that for small $\tau$, geometry is irrelevant to clique counts. [30] introduced a practical dimension estimator for GIRGs that has theoretical guarantees.

---

## Statistical-Computational Gaps and Low-Degree Lower Bounds

[5] proved that any degree-$D$ polynomial test for the spherical RGG requires $d \lesssim n^{3/(D+1)}$, establishing a polynomial hierarchy of increasingly hard detection problems. [28] made the gap explicit for planted dense cycles: the information-theoretic threshold and the low-degree polynomial threshold differ by a polynomial factor in $n$. [33] characterised the precise threshold in $d$ at which subsets of a Wishart matrix $X^\top X$ converge in total variation to independent Gaussians, with phase transitions whose boundaries depend on the graph structure of the examined subset.

---

## Latent Geometry Recovery and Embedding

The latent space model of [34] introduced the paradigm of modelling social networks via positions in a latent Euclidean space, with maximum likelihood and Bayesian inference procedures showing improved fit over stochastic blockmodels. [35] extended this to a mixture model for clustering. [24] and [25] proved consistency of spectral embedding for random dot product graphs.

[36] derived minimax estimation rates for pairwise latent distances in RGGs with unknown link function. [37] showed that Universal Singular Value Thresholding achieves the minimax error rate for noisy matrix estimation. [38] showed that a Jaccard-index filter recovers the metric structure of an ER-perturbed proximity graph within a factor of 2. [39] used the edge clique number to recover the shortest-path metric of an ER-perturbed RGG within a factor of 3 for a wider perturbation range than prior work.

[40] proved sufficient conditions for location recovery in sparse geometric random graphs over homogeneous metric spaces, and studied information percolation in a geometric branching random walk. [41] proved sufficient conditions for exact vertex correspondence recovery in correlated RGGs via the $k$-core estimator. [42] developed a practical method for estimating the latent dimension of real-world networks without requiring spatial embedding.

---

## Hidden Geometry in Real and Synthetic Networks

[43] showed that the self-similarity of scale-free networks under degree-thresholding renormalisation is explained by hidden metric spaces, with clustering serving as a topological reflection of the triangle inequality. [44] proved a converse: fixed expected degree and sufficiently strong clustering implies equivalence to an RGG on the real line.

[45] studied community detection on Euclidean RGGs, characterising recovery thresholds as a function of spatial intensity, dimension, and community separation. [46] introduced the Geometric Block Model, generalising the RGG to multi-community settings. [47] surveyed the statistical network analysis landscape across SBM, RGG, and graphon models. [48] proposed Markov Random Geometric Graphs (MRGGs) as a growth model for temporal dynamic networks, proving theoretical guarantees for nonparametric estimation of model parameters.

---

## Extensions to Simplicial Complexes

[49] constructed the *simplicial configuration model* as a principled null for simplicial complexes with fixed degree sequences, and developed an efficient MCMC sampler; Betti numbers of two out of three real-world systems tested were significantly inconsistent with the null. [50] developed goodness-of-fit tests for multi-parameter random simplicial complexes using simplex count statistics.

[51] characterised testing thresholds for sparse high-dimensional RGGs. [1] extended sandwiching results to yield sharp threshold results for monotone properties of simplicial complexes. [23] generalised spectral clustering to geometric simplicial complexes, and [22] established the spectral gap for the higher Laplacian on the Linial–Meshulam model.

---

## Stochastic Geometry: Probabilistic Foundations

[16] established the FCLT for subgraph counting processes over inhomogeneous Poisson point processes. [52, 53, 54] developed Malliavin-Stein machinery for Gaussian approximation of Poisson functionals, achieving presumably optimal rates of normal convergence for stabilising geometric functionals. [55] provided comprehensive Malliavin-Stein bounds for stochastic geometry applications.

[56] proved a large deviation principle for geometric component processes with the rate function expressed as relative entropy, deriving LDPs for persistent Betti numbers and Morse critical points as corollaries. [57] proved lower large deviations for geometric functionals without exponential moments via a sprinkling technique.

[58, 59] developed a comparison theory for point processes based on void probabilities and factorial moment measures, enabling bounds on percolation and coverage in geometric models. [60] defined face and cycle percolation on random Vietoris–Rips complexes, proving sharp phase transitions.

---

## Gaps and Open Questions

**The sparse regime.** The information-theoretic threshold for geometry detection in the sparse RGG $G(n,p,d)$ with $p = o(1)$ was conjectured by [2] but remains unproven. [8, 9, 51] made significant progress, but the precise boundary and matching achievability results are not fully established.

**Statistical-computational gaps.** The Fourier analysis of [5] and the planted dense cycle analysis of [28] make gaps precise in specific models, but a general theory connecting latent space geometry to the degree of polynomial needed for detection remains to be developed. It is unclear whether sum-of-squares or other algorithms can break through the $d \sim n^{3/D}$ barrier.

**Beyond spheres: hyperbolic geometry.** [13] demonstrated that triangle counts fail for hyperbolic detection and introduced weighted triangles as a fix, but rigorous detection thresholds for hyperbolic models are not established. Sub-tree count CLTs on hyperbolic RGGs [61] suggest that the statistical behaviour differs qualitatively from the Euclidean case, but no detection threshold theory exists.

**Recovery versus detection.** [28] closed the impossibility side for inner product recovery at $d \approx nh(p)$, but the sharp threshold for recovery and whether efficient algorithms achieve the information-theoretic threshold remain open.

**Simplicial geometry detection.** Rigorous detection thresholds for latent geometry in random simplicial complexes — the simplicial analogue of [2] — have not been established. The interplay between the higher-order spectral gap [22] and geometry detection in the simplicial setting is particularly unexplored.

**Community structure and geometry.** A unified framework characterising when geometric versus community structure dominates the detection signal — interpolating between the SBM and RGG regimes — is lacking.

**Unconditional computational lower bounds.** The low-degree framework provides convincing but ultimately conjectural hardness evidence. Unconditional lower bounds for geometry detection from cryptographic assumptions or average-case complexity remain an open challenge.

---

## References

1. Kiril Bangachev; Guy Bresler (2024). *Sandwiching Random Geometric Graphs and Erdos-Renyi with Applications: Sharp Thresholds, Robust Testing, and Enumeration*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2408.00995
2. Sébastien Bubeck; Jian Ding; Ronen Eldan; Miklós Z. Rácz (2014). *Testing for high‐dimensional geometry in random graphs*. Random Struct. Algorithms. https://doi.org/10.1002/rsa.20633
3. Luc Devroye; András György; Gábor Lugosi; Frederic Udina (2011). *High-Dimensional Random Geometric Graphs and their Clique Number*. Electronic Journal of Probability. https://doi.org/10.1214/ejp.v16-967
4. Elliot Paquette; Andrew Vander Werf (2021). *Random geometric graphs and the spherical Wishart matrix*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2110.10785
5. Kiril Bangachev; Guy Bresler (2024). *On the Fourier Coefficients of High-Dimensional Random Geometric Graphs*. Symposium on the Theory of Computing. https://doi.org/10.1145/3618260.3649676
6. Matthew Brennan; Guy Bresler; Brice Huang (2023). *Threshold for detecting high dimensional geometry in anisotropic random geometric graphs*. Random Structures and Algorithms. https://doi.org/10.1002/rsa.21178
7. Ronen Eldan; Dan Mikulincer (2017). [Title unknown; cited as conjecture source for detection in anisotropic RGGs. Not in corpus.]
8. Suqi Liu; Miklós Z. Rácz (2023). *Phase transition in noisy high-dimensional random geometric graphs*. Electronic Journal of Statistics. https://doi.org/10.1214/23-ejs2162
9. Suqi Liu; Miklós Z. Rácz (2023). *A probabilistic view of latent space graphs and phase transitions*. Bernoulli. https://doi.org/10.3150/22-bej1547
10. Oliver Baker; Carl P. Dettmann (2025). *Entropy of random geometric graphs in high and low dimensions*. Journal of Complex Networks. https://doi.org/10.1093/comnet/cnaf033
11. Kiril Bangachev; Guy Bresler (2023). *Random Algebraic Graphs and Their Convergence to Erdos-Renyi*. arXiv.org. https://doi.org/10.48550/arXiv.2305.04802
12. Kiril Bangachev; Guy Bresler (2025). *Random Algebraic Graphs and Their Convergence to ErdőS–Rényi*. Random Structures and Algorithms. https://doi.org/10.1002/rsa.21276
13. Riccardo Michielan; Nelly Litvak; Clara Stegehuis (2022). *Detecting hyperbolic geometry in networks: Why triangles are not enough*. Physical review. E. https://doi.org/10.1103/physreve.106.054303
14. Gianmarco Bet; Kay Bogerd; Rui M. Castro; Remco van der Hofstad (2021). *Detecting a botnet in a network*. Mathematical Statistics and Learning. https://doi.org/10.4171/msl/23
15. Gianmarco Bet; Riccardo Michielan; Clara Stegehuis (2025). *Localized geometry detection in scale-free random graphs*. Journal of Applied Probability. https://doi.org/10.1017/jpr.2025.10038
16. Takashi Owada (2017). *Functional central limit theorem for subgraph counting processes*. Electronic Journal of Probability. https://doi.org/10.1214/17-ejp30
17. Svante Janson; Andrzej Ruciński (2002). *The infamous upper tail*. Random Structures and Algorithms. https://doi.org/10.1002/rsa.10031
18. Milan Bradonjić; Will Perkins (2013). *On Sharp Thresholds in Random Geometric Graphs*. DROPS (Schloss Dagstuhl – Leibniz Center for Informatics). https://doi.org/10.4230/lipics.approx-random.2014.500
19. Barbara Brune; Jonathan Flossdorf; Carsten Jentsch (2024). *Goodness‐of‐fit testing based on graph functionals for homogeneous Erdös–Rényi graphs*. Scandinavian Journal of Statistics. https://doi.org/10.1111/sjos.12750
20. Ery Arias-Castro; Sébastien Bubeck; Gábor Lugosi (2015). *Detecting positive correlations in a multivariate sample*. Bernoulli. https://doi.org/10.3150/13-bej565
21. Paul Blackwell; Mark Edmondson-Jones; Jonathan Jordan (2006). *Spectra of adjacency matrices of random geometric graphs*.
22. Antti Knowles; Ron Rosenthal (2017). *Eigenvalue confinement and spectral gap for random simplicial complexes*. Random Structures and Algorithms. https://doi.org/10.1002/rsa.20710
23. Konstantin Avrachenkov; Andrei Bobu; Maximilien Dreveton (2021). *Higher-Order Spectral Clustering for Geometric Graphs*. Journal of Fourier Analysis and Applications. https://doi.org/10.1007/s00041-021-09825-2
24. Daniel L. Sussman; Minh Tang; Carey E. Priebe (2013). *Consistent Latent Position Estimation and Vertex Classification for Random Dot Product Graphs*. IEEE Transactions on Pattern Analysis and Machine Intelligence. https://doi.org/10.1109/tpami.2013.135
25. Minh Tang; Daniel L. Sussman; Carey E. Priebe (2013). *Universally consistent vertex classification for latent positions graphs*. The Annals of Statistics. https://doi.org/10.1214/13-aos1112
26. Shuangping Li; Tselil Schramm (2023). *Spectral clustering in the Gaussian mixture block model*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2305.00979
27. Vinay Jethava; Jacob Sznajdman; Chiranjib Bhattacharyya; Devdatt Dubhashi (2013). *Lovasz &amp;#x03D1;, SVMs and applications*. https://doi.org/10.1109/itw.2013.6691323
28. Cheng Mao; Alexander S. Wein; Shenduo Zhang (2024). *Information-Theoretic Thresholds for Planted Dense Cycles*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2402.00305
29. Cheng Mao; Shenduo Zhang (2024). *Impossibility of Latent Inner Product Recovery via Rate Distortion*. https://doi.org/10.1109/allerton63246.2024.10735333
30. Tobias Friedrich; Andreas Göbel; Maximilian Katzmann; Leon Schiller (2023). *Real-World Networks are Low-Dimensional: Theoretical and Practical Assessment*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2302.06357
31. Tobias Friedrich; Andreas Göbel; Maximilian Katzmann; Leon Schiller (2024). *Cliques in High-Dimensional Geometric Inhomogeneous Random Graphs*. SIAM Journal on Discrete Mathematics. https://doi.org/10.1137/23m157394x
32. Riccardo Michielan; Clara Stegehuis (2021). *Cliques in geometric inhomogeneous random graphs*. Journal of Complex Networks. https://doi.org/10.1093/comnet/cnac002
33. Matthew Brennan; Guy Bresler; Brice Huang (2021). *De Finetti-Style Results for Wishart Matrices: Combinatorial Structure and Phase Transitions*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2103.14011
34. Peter D. Hoff; Adrian E. Raftery; Mark S. Handcock (2002). *Latent Space Approaches to Social Network Analysis*. Journal of the American Statistical Association. https://doi.org/10.1198/016214502388618906
35. Mark S. Handcock; Adrian E. Raftery; Jeremy Tantrum (2007). *Model-Based Clustering for Social Networks*. Journal of the Royal Statistical Society Series A (Statistics in Society). https://doi.org/10.1111/j.1467-985x.2007.00471.x
36. Ernesto Araya; Yohann de Castro (2019). *Latent Distance Estimation for Random Geometric Graphs*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1909.06841
37. Sourav Chatterjee (2014). *Matrix estimation by Universal Singular Value Thresholding*. The Annals of Statistics. https://doi.org/10.1214/14-aos1272
38. Srinivasan Parthasarathy; David Sivakoff; Minghao Tian; Yusu Wang (2017). *A Quest to Unravel the Metric Structure Behind Perturbed Networks*. DROPS (Schloss Dagstuhl – Leibniz Center for Informatics). https://doi.org/10.4230/lipics.socg.2017.53
39. Matthew Kahle; Minghao Tian; Yusu Wang (2018). *Local Cliques in ER-Perturbed Random Geometric Graphs*. DROPS (Schloss Dagstuhl – Leibniz Center for Informatics). https://doi.org/10.4230/lipics.isaac.2019.29
40. Ronen Eldan; Dan Mikulincer; Hester Pieters (2022). *Community detection and percolation of information in a geometric setting*. Combinatorics Probability Computing. https://doi.org/10.1017/s0963548322000098
41. Miklós Z. Rácz; Anirudh Sridhar (2023). *Matching Correlated Inhomogeneous Random Graphs using the k-core Estimator*. https://doi.org/10.1109/isit54713.2023.10206932
42. Pedro Almagro; Marián Boguñá; M. Ángeles Serrano (2022). *Detecting the ultra low dimensionality of real networks*. Nature Communications. https://doi.org/10.1038/s41467-022-33685-z
43. M. Ángeles Serrano; Dmitri Krioukov; Marián Boguñá (2008). *Self-Similarity of Complex Networks and Hidden Metric Spaces*. Physical Review Letters. https://doi.org/10.1103/physrevlett.100.078701
44. Dmitri Krioukov (2016). *Clustering Implies Geometry in Networks*. Physical Review Letters. https://doi.org/10.1103/physrevlett.116.208302
45. Abishek Sankararaman; François Baccelli (2017). *Community Detection on Euclidean Random Graphs*. https://doi.org/10.1109/allerton.2017.8262780
46. Sainyam Galhotra; Arya Mazumdar; Soumyabrata Pal; Barna Saha (2018). *The Geometric Block Model*. Proceedings of the AAAI Conference on Artificial Intelligence. https://doi.org/10.1609/aaai.v32i1.11905
47. Miklós Z. Rácz; Sébastien Bubeck (2017). *Basic models and questions in statistical network analysis*. Statistics Surveys. https://doi.org/10.1214/17-ss117
48. Quentin Duchemin; Yohann de Castro (2022). *Markov random geometric graph, MRGG: A growth model for temporal dynamic networks*. Electronic Journal of Statistics. https://doi.org/10.1214/21-ejs1969
49. Jean-Gabriel Young; Giovanni Petri; Francesco Vaccarino; Alice Patania (2017). *Construction of and efficient sampling from the simplicial configuration model*. Physical review. E. https://doi.org/10.1103/physreve.96.032312
50. Tadas Temčinas; Vidit Nanda; Gesine Reinert (2025). *Goodness-of-fit via count statistics in dense random simplicial complexes*. Foundations of Data Science. https://doi.org/10.3934/fods.2025013
51. Siqi Liu; Sidhanth Mohanty; Tselil Schramm; Elizabeth Yang (2022). *Testing thresholds for high-dimensional sparse random geometric graphs*. Symposium on the Theory of Computing. https://doi.org/10.1145/3519935.3519989
52. Raphaël Lachièze-Rey; Giovanni Peccati (2013). *Fine Gaussian fluctuations on the Poisson space, I: contractions, cumulants and geometric random graphs*. Electronic Journal of Probability. https://doi.org/10.1214/ejp.v18-2104
53. Raphaël Lachièze-Rey; Matthias Schulte; J. E. Yukich (2019). *Normal approximation for stabilizing functionals*. The Annals of Applied Probability. https://doi.org/10.1214/18-aap1405
54. Raphaël Lachièze-Rey; Giovanni Peccati; Xiaochuan Yang (2022). *Quantitative two-scale stabilization on the Poisson space*. The Annals of Applied Probability. https://doi.org/10.1214/21-aap1768
55. Matthias Schulte (2013). *Malliavin-Stein Method in Stochastic Geometry*. osnaDocs (Osnabrück University).
56. Christian Hirsch; Takashi Owada (2023). *Large deviation principle for geometric and topological functionals and associated point processes*. The Annals of Applied Probability. https://doi.org/10.1214/22-aap1914
57. Christian Hirsch; Daniel Willhalm (2024). *Lower large deviations for geometric functionals in sparse, critical and dense regimes*. Latin American Journal of Probability and Mathematical Statistics. https://doi.org/10.30757/alea.v21-38
58. Bartłomiej Błaszczyszyn; D. Yogeshwaran (2013). *Clustering and percolation of point processes*. Electronic Journal of Probability. https://doi.org/10.1214/ejp.v18-2468
59. Bartłomiej Błaszczyszyn; D. Yogeshwaran (2014). *On Comparison of Clustering Properties of Point Processes*. Advances in Applied Probability. https://doi.org/10.1239/aap/1396360100
60. Christian Hirsch; Daniel Valesin (2025). *Face and cycle percolation*. Journal of Applied and Computational Topology. https://doi.org/10.1007/s41468-025-00202-2
61. Takashi Owada; D. Yogeshwaran (2022). *Sub-tree counts on hyperbolic random geometric graphs*. Advances in Applied Probability. https://doi.org/10.1017/apr.2022.1