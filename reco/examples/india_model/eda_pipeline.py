import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import io

# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format to show only the message
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EDAPipeline:
    def __init__(self, data_path):
        """
        Initialize the EDA pipeline with a path to the dataset.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = Path(data_path)
        self.df = None
        
        plt.style.use('default')
        sns.set_theme() 
        
    def load_data(self):
        """Load and perform initial data exploration"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"\nDataset Shape: {self.df.shape}")
            
            # Display basic information more cleanly
            logger.info("\nDataset Info:")
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            logger.info(buffer.getvalue())
            
            logger.info("\nFirst few rows:")
            with pd.option_context('display.max_columns', None):
                logger.info(f"\n{self.df.head().to_string()}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def check_data_quality(self):
        """Perform data quality checks"""
        if self.df is None:
            logger.error("Data not loaded. Please run load_data() first.")
            return

        # Check missing values - only show if there are any
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        if not missing_values.empty:
            logger.info(f"\nMissing Values: {missing_values}")
        # Check duplicates - only show if there are any
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"\nDuplicate Rows: {duplicates}")

        # Statistical summary with cleaner formatting
        logger.info("\nStatistical Summary:")
        with pd.option_context('display.float_format', '{:.2f}'.format):
            logger.info(f"\n{self.df.describe().to_string()}")

    def analyze_categorical(self, columns=None):
        """
        Analyze categorical variables
        
        Args:
            columns (list, optional): List of categorical columns to analyze. 
                                    If None, analyzes all categorical columns.
        """
        if self.df is None:
            logger.error("Data not loaded. Please run load_data() first.")
            return

        # If no columns specified, use all categorical columns
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        elif not isinstance(columns, (list, np.ndarray, pd.Index)):
            columns = [columns]

        # Filter out non-existent columns
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            logger.error("No valid columns found for analysis")
            return

        # Calculate grid dimensions
        n_cols = min(3, len(valid_columns))  # Maximum 3 columns in grid
        n_rows = (len(valid_columns) + n_cols - 1) // n_cols
        
        # Create figure for bar plots
        fig = plt.figure(figsize=(7*n_cols, 5*n_rows))
        fig.suptitle('Categorical Distributions', y=1.02, fontsize=16)

        # Plot each column
        for idx, column in enumerate(valid_columns):
            plt.subplot(n_rows, n_cols, idx + 1)
            
            # Get value counts and sort for better visualization
            value_counts = self.df[column].value_counts()
            
            # If there are too many categories, limit to top 10
            if len(value_counts) > 10:
                logger.info(f"\nNote: {column} has {len(value_counts)} categories. Showing top 10.")
                value_counts = value_counts.head(10)
            
            # Create bar plot
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f'Distribution of {column}')
            
            # Rotate labels if they're too long
            if value_counts.index.str.len().max() > 10:
                plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(value_counts.values):
                plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.show()

        # Print frequency distributions
        logger.info("\nFrequency distributions (proportions):")
        for column in valid_columns:
            freq_dist = self.df[column].value_counts(normalize=True)
            logger.info(f"\n{column}:")
            logger.info(f"{freq_dist.to_string()}")
            
            # Print additional statistics
            logger.info(f"Number of unique values: {self.df[column].nunique()}")
            logger.info(f"Most common value: {self.df[column].mode().iloc[0]} "
                       f"({self.df[column].value_counts(normalize=True).iloc[0]:.2%})")

    def analyze_numerical(self, columns=None):
        """
        Analyze numerical variables
        
        Args:
            columns (list, optional): List of numerical columns to analyze. 
                                    If None, analyzes all numerical columns.
        """
        if self.df is None:
            logger.error("Data not loaded. Please run load_data() first.")
            return

        # If no columns specified, use all numerical columns
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        elif not isinstance(columns, (list, np.ndarray, pd.Index)):
            columns = [columns]

        # Filter out non-existent columns
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            logger.error("No valid columns found for analysis")
            return

        # Calculate grid dimensions
        n_cols = min(3, len(valid_columns))  # Maximum 3 columns in grid
        n_rows = (len(valid_columns) + n_cols - 1) // n_cols
        
        # Create subplots for histograms
        fig_hist = plt.figure(figsize=(6*n_cols, 4*n_rows))
        fig_hist.suptitle('Distribution Plots', y=1.02, fontsize=16)
        
        # Create subplots for box plots
        fig_box = plt.figure(figsize=(6*n_cols, 4*n_rows))
        fig_box.suptitle('Box Plots', y=1.02, fontsize=16)

        # Plot each column
        for idx, column in enumerate(valid_columns):
            # Histogram with KDE
            plt.figure(fig_hist.number)
            plt.subplot(n_rows, n_cols, idx + 1)
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45)
            
            # Box plot
            plt.figure(fig_box.number)
            plt.subplot(n_rows, n_cols, idx + 1)
            sns.boxplot(y=self.df[column])
            plt.title(f'Box Plot of {column}')
        
        # Adjust layout
        plt.figure(fig_hist.number)
        plt.tight_layout()
        plt.show()
        
        plt.figure(fig_box.number)
        plt.tight_layout()
        plt.show()

        # Print summary statistics
        logger.info("\nSummary statistics:")
        with pd.option_context('display.float_format', '{:.2f}'.format):
            summary = self.df[valid_columns].describe()
            logger.info(f"\n{summary.to_string()}")

    def correlation_analysis(self):
        """Perform correlation analysis on numerical columns"""
        if self.df is None:
            logger.error("Data not loaded. Please run load_data() first.")
            return

        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) < 2:
            logger.warning("Not enough numerical columns for correlation analysis")
            return

        # Calculate correlation matrix
        correlation_matrix = self.df[numerical_cols].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

        # Print strong correlations using logger instead of print
        logger.info("\nStrong correlations (|correlation| > 0.5):")
        strong_corr = (correlation_matrix.abs() > 0.5) & (correlation_matrix != 1.0)
        strong_correlations = []
        for col in correlation_matrix.columns:
            strong_pairs = strong_corr[col][strong_corr[col]].index
            for pair in strong_pairs:
                strong_correlations.append(
                    f"{col} - {pair}: {correlation_matrix.loc[col, pair]:.3f}"
                )
        if strong_correlations:
            logger.info("\n" + "\n".join(strong_correlations))
        else:
            logger.info("\nNo strong correlations found")

def main():
    # Add a title banner for better organization
    logger.info("\n" + "="*50)
    logger.info("Exploratory Data Analysis Pipeline")
    logger.info("="*50 + "\n")

    # Example usage
    data_path = "path_to_your_data.csv"  # Replace with actual path
    eda = EDAPipeline(data_path)
    
    # Run the analysis
    try:
        # Load and explore data
        eda.load_data()
        
        # Check data quality
        eda.check_data_quality()
        
        # Analyze all categorical columns together
        eda.analyze_categorical()  # Will analyze all categorical columns
        
        # Or analyze specific columns
        # eda.analyze_categorical(['State', 'Type'])
        
        # Analyze all numerical columns together
        eda.analyze_numerical()
        
        # Perform correlation analysis
        eda.correlation_analysis()
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 