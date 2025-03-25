import os
import zipfile
# from Bio import SeqIO
# from Bio.Seq import Seq
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


class RNAProteinBindingSiteAnalysis:
    def __init__(self, protein_name):
        """
        Initialize the analysis with the selected protein.

        Parameters:
        -----------
        protein_name : str
            Name of the protein to analyze (either 'hnrnpc' or 'hnrnpa2b1')
        """
        self.protein_name = protein_name.lower()
        if self.protein_name not in ['hnrnpc', 'hnrnpa2b1']:
            raise ValueError("Protein name must be either 'hnrnpc' or 'hnrnpa2b1'")

        self.expected_motif = None
        self.expected_motif_fshape = None
        self.motif_length = None
        self.binding_sites_data = []
        self.search_data = []
        self.motifs = {}  # Dictionary to store motifs of different lengths
        self.clusters = {}  # Dictionary to store clustering results
        self.consensus_motifs = {}  # Dictionary to store consensus motifs
        self.search_results = []  # List to store search results

    def load_expected_motif(self, file_path=None):
        """
        Load the expected motif from the text file.

        Parameters:
        -----------
        file_path : str, optional
            Path to the file containing the expected motif. If None, uses default naming.
        """
        if file_path is None:
            file_path = f"{self.protein_name}_expected_pattern.txt"

        try:
            # Read the expected motif file
            motif_data = pd.read_csv(file_path, sep='\t', header=None, names=['fSHAPE', 'Base'])

            # Store the expected motif data
            self.expected_motif = ''.join(motif_data['Base'].values)
            self.expected_motif_fshape = motif_data['fSHAPE'].values
            self.motif_length = len(self.expected_motif)

            print(f"Expected motif loaded: {self.expected_motif} (length: {self.motif_length})")
            return True
        except Exception as e:
            print(f"Error loading expected motif: {e}")
            raise e
            return False

    def extract_binding_sites_data(self, zip_file_path=None):
        """
        Extract binding sites data from the zip archive.

        Parameters:
        -----------
        zip_file_path : str, optional
            Path to the zip file containing binding sites data. If None, uses default naming.
        """
        if zip_file_path is None:
            zip_file_path = f"{self.protein_name}_binding_sites_fshape.zip"

        try:
            # Extract all files from the zip archive
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Create a temporary directory to extract files
                dir = f"{self.protein_name}_binding_sites_fshape"
                os.makedirs(dir, exist_ok=True)
                zip_ref.extractall()

                # Process each file in the directory
                for file_name in os.listdir(dir):
                    file_path = os.path.join(dir, file_name)

                    # Read the file
                    try:
                        # Check if file has 2 or 3 columns
                        with open(file_path, 'r') as f:
                            first_line = f.readline().strip()
                            num_columns = len(first_line.split('\t'))

                        if num_columns == 2:
                            data = pd.read_csv(file_path, sep='\t', header=None, names=['fSHAPE', 'Base'])
                        elif num_columns == 3:
                            data = pd.read_csv(file_path, sep='\t', header=None, names=['fSHAPE', 'Base', 'SHAPE'])
                        else:
                            print(f"Warning: Unexpected number of columns ({num_columns}) in file {file_name}")
                            continue

                        # Store the data
                        self.binding_sites_data.append(
                            {
                                'file_name': file_name,
                                'data': data
                            }
                        )
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

            print(f"Loaded {len(self.binding_sites_data)} binding site files")
            return True
        except Exception as e:
            print(f"Error extracting binding sites data: {e}")
            return False

    def extract_search_data(self, zip_file_path=None):
        """
        Extract search data from the zip archive.

        Parameters:
        -----------
        zip_file_path : str, optional
            Path to the zip file containing search data. If None, uses default naming.
        """
        if zip_file_path is None:
            zip_file_path = f"{self.protein_name}_search_fshape.zip"

        try:
            # Extract all files from the zip archive
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Create a temporary directory to extract files
                dir = f"{self.protein_name}_search_fshape"
                os.makedirs(dir, exist_ok=True)
                zip_ref.extractall()

                # Process each file in the directory
                for file_name in os.listdir(dir):
                    file_path = os.path.join(dir, file_name)

                    # Read the file
                    try:
                        # Check if file has 2 or 3 columns
                        with open(file_path, 'r') as f:
                            first_line = f.readline().strip()
                            num_columns = len(first_line.split('\t'))

                        if num_columns == 2:
                            data = pd.read_csv(file_path, sep='\t', header=None, names=['fSHAPE', 'Base'])
                        elif num_columns == 3:
                            data = pd.read_csv(file_path, sep='\t', header=None, names=['fSHAPE', 'Base', 'SHAPE'])
                        else:
                            print(f"Warning: Unexpected number of columns ({num_columns}) in file {file_name}")
                            continue

                        # Store the data
                        self.search_data.append(
                            {
                                'file_name': file_name,
                                'data': data
                            }
                        )
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

            print(f"Loaded {len(self.search_data)} search files")
            return True
        except Exception as e:
            print(f"Error extracting search data: {e}")
            raise e
            return False

    def extract_promising_motifs(self):
        """
        Extract promising motifs from binding sites data.
        """
        if self.motif_length is None:
            print("Error: Expected motif not loaded")
            return False

        # Define the lengths of motifs to extract
        motif_lengths = [self.motif_length, self.motif_length + 1, self.motif_length + 2]

        for length in motif_lengths:
            self.motifs[length] = []

            for binding_site in self.binding_sites_data:
                data = binding_site['data']

                # Skip if data is too short
                if len(data) < length:
                    continue

                # Extract continuous fragments of specified length
                for i in range(len(data) - length + 1):
                    fragment = data.iloc[i:i + length]

                    # Skip fragments with NaN values
                    if fragment['fSHAPE'].isna().any():
                        continue

                    # Check if at least one nucleotide has fSHAPE > 1.0
                    if (fragment['fSHAPE'] > 1.0).any():
                        sequence = ''.join(fragment['Base'].values)
                        fshape_values = fragment['fSHAPE'].values

                        self.motifs[length].append(
                            {
                                'sequence': sequence,
                                'fshape': fshape_values,
                                'file_name': binding_site['file_name'],
                                'position': i
                            }
                        )

            print(f"Extracted {len(self.motifs[length])} promising motifs of length {length}")

        return True

    def perform_clustering(self):
        """
        Perform clustering on extracted motifs.
        """
        if not self.motifs:
            print("Error: No motifs extracted")
            return False

        for length, motifs in self.motifs.items():
            if not motifs:
                print(f"No motifs of length {length} to cluster")
                continue

            # Prepare data for clustering
            X = np.array([motif['fshape'] for motif in motifs])

            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform K-means clustering
            kmeans_results = self._perform_kmeans(X_scaled, length)

            # Perform DBSCAN clustering
            dbscan_results = self._perform_dbscan(X_scaled, length)

            # Store clustering results
            self.clusters[length] = {
                'kmeans': kmeans_results,
                'dbscan': dbscan_results
            }

        return True

    def _perform_kmeans(self, X, length):
        """
        Perform K-means clustering.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to cluster
        length : int
            Length of motifs

        Returns:
        --------
        dict
            Clustering results
        """
        # Determine number of clusters based on data size
        n_clusters = min(10, max(3, len(X) // 5))

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Count elements in each cluster
        cluster_counts = np.bincount(labels)

        # Get the three most abundant clusters
        top_clusters = np.argsort(-cluster_counts)[:3]

        # Store results
        results = {
            'labels': labels,
            'cluster_counts': cluster_counts,
            'top_clusters': top_clusters
        }

        print(f"K-means clustering for length {length}: {n_clusters} clusters created")
        print(f"Top 3 clusters: {top_clusters} with counts: {cluster_counts[top_clusters]}")

        return results

    def _perform_dbscan(self, X, length):
        """
        Perform DBSCAN clustering.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to cluster
        length : int
            Length of motifs

        Returns:
        --------
        dict
            Clustering results
        """
        # Determine epsilon based on data dimensionality
        eps = 0.5 * np.sqrt(length)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(X)

        # Count elements in each cluster (excluding noise points with label -1)
        unique_labels = np.unique(labels)
        cluster_counts = np.array([np.sum(labels == label) for label in unique_labels if label != -1])

        # Get the three most abundant clusters (excluding noise)
        if len(cluster_counts) > 0:
            top_cluster_indices = np.argsort(-cluster_counts)[:min(3, len(cluster_counts))]
            top_clusters = np.array([label for label in unique_labels if label != -1])[top_cluster_indices]
        else:
            top_clusters = np.array([])

        # Store results
        results = {
            'labels': labels,
            'cluster_counts': cluster_counts,
            'top_clusters': top_clusters
        }

        print(
            f"DBSCAN clustering for length {length}: {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters created"
        )
        if len(top_clusters) > 0:
            print(f"Top clusters: {top_clusters} with counts: {[np.sum(labels == label) for label in top_clusters]}")
        else:
            print("No significant clusters found with DBSCAN")

        return results

    def determine_consensus_motifs(self):
        """
        Determine consensus motifs from clustering results.
        """
        if not self.clusters:
            print("Error: No clustering results available")
            return False

        for length, clustering_results in self.clusters.items():
            self.consensus_motifs[length] = {}

            for method, results in clustering_results.items():
                consensus_motifs = []

                for cluster_id in results['top_clusters']:
                    # Get motifs in this cluster
                    if method == 'kmeans':
                        cluster_motifs = [self.motifs[length][i] for i in range(len(self.motifs[length]))
                                          if results['labels'][i] == cluster_id]
                    else:  # DBSCAN
                        cluster_motifs = [self.motifs[length][i] for i in range(len(self.motifs[length]))
                                          if results['labels'][i] == cluster_id]

                    # Check if cluster has minimum required size
                    if len(cluster_motifs) < 3:
                        continue

                    # Extract fSHAPE profiles for motifs in this cluster
                    profiles = np.array([motif['fshape'] for motif in cluster_motifs])

                    # Determine consensus profile (average of all profiles)
                    consensus_profile = np.mean(profiles, axis=0)

                    # Determine consensus sequence
                    sequences = [motif['sequence'] for motif in cluster_motifs]
                    consensus_sequence = self._determine_consensus_sequence(sequences)

                    consensus_motifs.append(
                        {
                            'cluster_id': cluster_id,
                            'size': len(cluster_motifs),
                            'consensus_sequence': consensus_sequence,
                            'consensus_profile': consensus_profile
                        }
                    )

                self.consensus_motifs[length][method] = consensus_motifs
                print(f"Determined {len(consensus_motifs)} consensus motifs for length {length} using {method}")

        return True

    def _determine_consensus_sequence(self, sequences):
        """
        Determine consensus sequence from a list of sequences.

        Parameters:
        -----------
        sequences : list
            List of sequences

        Returns:
        --------
        str
            Consensus sequence
        """
        if not sequences:
            return ""

        # Check if all sequences have the same length
        length = len(sequences[0])
        if not all(len(seq) == length for seq in sequences):
            print("Warning: Not all sequences have the same length. Using the first sequence length.")

        # Initialize consensus sequence
        consensus = ""

        # Determine the most common nucleotide at each position
        for i in range(length):
            nucleotides = [seq[i] if i < len(seq) else 'N' for seq in sequences]
            counts = Counter(nucleotides)
            most_common = counts.most_common(1)[0][0]
            consensus += most_common

        return consensus

    def identify_consensus_motifs_with_stumpy(self):
        """
        Identify consensus motifs using STUMPY library based on fSHAPE profiles
        from the most abundant clusters.
        """
        if not self.clusters or not self.consensus_motifs:
            print("Error: No clustering or consensus motifs available")
            return False

        # Dictionary to store refined consensus motifs
        self.stumpy_consensus_motifs = {}

        for length, clustering_results in self.clusters.items():
            self.stumpy_consensus_motifs[length] = {}

            for method, cluster_data in clustering_results.items():
                refined_motifs = []

                # Get the three most abundant clusters
                for cluster_id in cluster_data['top_clusters']:
                    # Get motifs in this cluster
                    if method == 'kmeans':
                        cluster_motifs = [self.motifs[length][i] for i in range(len(self.motifs[length]))
                                          if cluster_data['labels'][i] == cluster_id]
                    else:  # DBSCAN
                        cluster_motifs = [self.motifs[length][i] for i in range(len(self.motifs[length]))
                                          if cluster_data['labels'][i] == cluster_id]

                    # Skip if cluster size is less than 3
                    if len(cluster_motifs) < 3:
                        continue

                    # Extract fSHAPE profiles for motifs in this cluster
                    profiles = np.array([motif['fshape'] for motif in cluster_motifs])

                    # Use a simpler approach to find the consensus motif
                    if len(profiles) >= 3:
                        # Calculate the average profile as the consensus
                        consensus_profile = np.mean(profiles, axis=0)

                        # Find the profile closest to the consensus
                        min_dist = float('inf')
                        best_motif_idx = 0

                        for i, profile in enumerate(profiles):
                            # Calculate Euclidean distance to consensus
                            dist = np.sum((profile - consensus_profile) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_motif_idx = i

                        best_motif_profile = profiles[best_motif_idx]
                        best_motif_sequence = cluster_motifs[best_motif_idx]['sequence']

                        refined_motifs.append(
                            {
                                'cluster_id': cluster_id,
                                'size': len(cluster_motifs),
                                'sequence': best_motif_sequence,
                                'profile': best_motif_profile,
                                'motif_source': cluster_motifs[best_motif_idx]['file_name'],
                                'motif_position': cluster_motifs[best_motif_idx]['position']
                            }
                        )

                self.stumpy_consensus_motifs[length][method] = refined_motifs
                print(
                    f"Identified {len(refined_motifs)} consensus motifs for length {length} using {method} with STUMPY"
                )

        return True

    def search_transcripts(self):
        """
        Search for motifs in transcripts based on the consensus motifs
        and expected motif profiles.
        """
        if not hasattr(self, 'stumpy_consensus_motifs') and self.expected_motif_fshape is None:
            print("Error: No consensus motifs or expected motif available for search")
            return False

        if not self.search_data:
            print("Error: No search data available")
            return False

        # List to store search results
        self.search_results = []

        # Add expected motif to the search queries
        search_queries = [{
            'profile': self.expected_motif_fshape,
            'sequence': self.expected_motif,
            'type': 'expected',
            'length': len(self.expected_motif),
            'source': 'expected_motif'
        }]

        # Add consensus motifs to the search queries
        if hasattr(self, 'stumpy_consensus_motifs'):
            for length, method_results in self.stumpy_consensus_motifs.items():
                for method, motifs in method_results.items():
                    for motif in motifs:
                        search_queries.append(
                            {
                                'profile': motif['profile'],
                                'sequence': motif['sequence'],
                                'type': f"{method}_consensus",
                                'length': length,
                                'source': f"{motif['motif_source']}:{motif['motif_position']}"
                            }
                        )

        print(f"Searching for {len(search_queries)} motifs in {len(self.search_data)} transcripts...")

        # Search each transcript for each motif
        for transcript in self.search_data:
            transcript_data = transcript['data']
            transcript_fshape = transcript_data['fSHAPE'].values
            transcript_sequence = ''.join(transcript_data['Base'].values)

            # Skip if data contains NaN values
            if np.isnan(transcript_fshape).any():
                nan_cleaned_fshape = np.copy(transcript_fshape)
                nan_indices = np.isnan(nan_cleaned_fshape)
                nan_cleaned_fshape[nan_indices] = 0  # Replace NaN with 0 for search
            else:
                nan_cleaned_fshape = transcript_fshape

            # Search for each motif
            for query in search_queries:
                query_profile = query['profile']
                query_sequence = query['sequence']
                query_length = len(query_profile)

                # Skip if transcript is shorter than the query
                if len(nan_cleaned_fshape) < query_length:
                    continue

                # Custom search implementation without using STUMPY's query parameter
                for i in range(len(nan_cleaned_fshape) - query_length + 1):
                    segment = nan_cleaned_fshape[i:i + query_length]

                    # Skip if the segment contains NaN values
                    if np.isnan(segment).any():
                        continue

                    # Calculate Euclidean distance
                    distance = np.sqrt(np.sum((segment - query_profile) ** 2))

                    # Normalize the distance
                    zned = distance / query_length

                    # Get the corresponding sequence
                    match_sequence = transcript_sequence[i:i + query_length]

                    # Calculate sequence similarity
                    seq_similarity = sum(1 for a, b in zip(match_sequence, query_sequence) if a == b) / query_length

                    # Calculate average similarity
                    avg_similarity = (1 - zned + seq_similarity) / 2

                    # Set a threshold for matches
                    threshold = 0.6  # Adjust this value as needed

                    if avg_similarity > threshold:
                        self.search_results.append(
                            {
                                'query_type': query['type'],
                                'query_source': query['source'],
                                'query_sequence': query_sequence,
                                'match_sequence': match_sequence,
                                'transcript': transcript['file_name'],
                                'position_start': i,
                                'position_end': i + query_length - 1,
                                'znEd': zned,
                                'ssf': seq_similarity,
                                'aS': avg_similarity
                            }
                        )

        # Sort results by aS in non-decreasing order
        self.search_results.sort(key=lambda x: x['aS'])

        print(f"Found {len(self.search_results)} matches across all transcripts")
        return True

    def generate_search_results_table(self):
        """
        Generate a table with search results.
        """
        if not self.search_results:
            print("Error: No search results available")
            return None

        # Create DataFrame from search results
        results_df = pd.DataFrame(
            self.search_results, columns=[
                'query_type', 'query_source', 'query_sequence', 'match_sequence',
                'transcript', 'position_start', 'position_end', 'znEd', 'ssf', 'aS'
            ]
        )

        # Format the position range
        results_df['position_range'] = results_df.apply(
            lambda row: f"{row['position_start']}-{row['position_end']}", axis=1
        )

        # Select and reorder columns for the final table
        final_table = results_df[[
            'query_type', 'query_sequence', 'match_sequence', 'position_range',
            'transcript', 'znEd', 'ssf', 'aS'
        ]]

        # Rename columns for better readability
        final_table.columns = [
            'Motif Type', 'Query Sequence', 'Match Sequence', 'Position Range',
            'Transcript', 'znEd', 'ssf', 'aS'
        ]

        return final_table

    def save_search_results(self, file_path=None):
        """
        Save search results to a CSV file.

        Parameters:
        -----------
        file_path : str, optional
            Path to save the results. If None, uses default naming.
        """
        if file_path is None:
            file_path = f"{self.protein_name}_search_results.csv"

        table = self.generate_search_results_table()
        if table is not None:
            table.to_csv(file_path, index=False)
            print(f"Search results saved to {file_path}")
            return True
        return False

    def visualize_top_matches(self, n=5):
        """
        Visualize top n matches based on aS score.

        Parameters:
        -----------
        n : int, optional
            Number of top matches to visualize. Default is 5.
        """
        if not self.search_results:
            print("Error: No search results available")
            return False

        # Sort results by aS in decreasing order and get top n
        top_matches = sorted(self.search_results, key=lambda x: x['aS'], reverse=True)[:n]

        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
        if n == 1:
            axes = [axes]

        for i, match in enumerate(top_matches):
            ax = axes[i]

            # Find the original transcript data
            transcript_data = None
            for transcript in self.search_data:
                if transcript['file_name'] == match['transcript']:
                    transcript_data = transcript['data']
                    break

            if transcript_data is None:
                continue

            # Extract the region of interest with some context
            context_size = 10
            start_idx = max(0, match['position_start'] - context_size)
            end_idx = min(len(transcript_data), match['position_end'] + context_size + 1)

            region_data = transcript_data.iloc[start_idx:end_idx]
            region_fshape = region_data['fSHAPE'].values
            region_sequence = ''.join(region_data['Base'].values)

            # Highlight the matched region
            match_start_rel = match['position_start'] - start_idx
            match_end_rel = match['position_end'] - start_idx + 1

            # Plot fSHAPE values
            ax.plot(region_fshape, 'b-', label='fSHAPE')
            ax.axvspan(match_start_rel, match_end_rel, alpha=0.2, color='red')

            # Add sequence as x-tick labels
            ax.set_xticks(range(len(region_sequence)))
            ax.set_xticklabels(list(region_sequence), rotation=0, fontsize=8)

            ax.set_title(f"Match {i + 1}: {match['match_sequence']} in {match['transcript']} (aS={match['aS']:.3f})")
            ax.set_ylabel('fSHAPE')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.protein_name}_top_matches.png")
        plt.show()

        return True


if __name__ == '__main__':
    # Initialize the analysis
    analysis = RNAProteinBindingSiteAnalysis('hnrnpc')

    # Load the expected motif
    analysis.load_expected_motif(
        file_path="Lab1/AIBI-lab-01-data/RBP-footprinting-data/HNRNPC/hnrnpc_expected_pattern.txt"
    )

    # Extract binding sites data
    analysis.extract_binding_sites_data(
        zip_file_path="Lab1/AIBI-lab-01-data/RBP-footprinting-data/HNRNPC/hnrnpc_binding_sites_fshape.zip"
    )

    # Extract search data
    analysis.extract_search_data(
        zip_file_path="Lab1/AIBI-lab-01-data/RBP-footprinting-data/HNRNPC/hnrnpc_search_fshape.zip"
    )

    # Extract promising motifs
    analysis.extract_promising_motifs()

    # Perform clustering
    analysis.perform_clustering()

    # Determine consensus motifs
    analysis.determine_consensus_motifs()

    # Identify consensus motifs with STUMPY
    analysis.identify_consensus_motifs_with_stumpy()

    # Search transcripts for motifs
    analysis.search_transcripts()

    # Generate and save search results
    analysis.save_search_results()

    # Visualize top matches
    analysis.visualize_top_matches(n=5)
