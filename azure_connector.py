import pandas as pd
import logging
import time
from typing import Dict, Union, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import lit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureConnector:
    """
    Simple class to write data to Azure SQL tables from Databricks.
    """
    def __init__(self, server: str, database: str, username: str, password: str):
        """
        Initialize with Azure SQL connection parameters.
        
        Args:
            server: Azure SQL server name
            database: Database name
            username: SQL username
            password: SQL password
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self._spark = None
    
    def get_spark(self) -> SparkSession:
        """Get the Spark session"""
        if self._spark is None:
            self._spark = SparkSession.builder.getOrCreate()
        return self._spark
    
    def get_connection_properties(self) -> tuple:
        """Get JDBC connection properties"""
        # Create a simple JDBC URL
        jdbc_url = f"jdbc:sqlserver://{self.server}:1433;databaseName={self.database}"
        
        # Connection properties
        connection_properties = {
            "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
            "user": self.username,
            "password": self.password,
            "encrypt": "true",
            "trustServerCertificate": "true"
        }
        
        return jdbc_url, connection_properties
    
    def read_dataframe(self):
        """
        Read a Dataframe from Azure SQL to be used in Prophet AI

        Args:
            table_name: Name of the table to read from Azure SQL
        """
        spark = self.get_spark()
        jdbc_url, connection_properties = self.get_connection_properties()

        query = """
                    SELECT ds, sum(y) as y
                    FROM DB.DailySalesDataset
                    GROUP BY ds
                    """
                    
        logger.info (f"Reading data from Azure SQL")
        # Read the data from Azure SQL
        df = spark.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("query", query) \
            .option("user", self.username) \
            .option("password", self.password) \
            .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
            .load()

        if df.rdd.isEmpty():
            logger.warning("No data read from Azure SQL.")
        else:
            logger.info("Successfully read data from Azure SQL.")

        return df.toPandas()
  
    def write_dataframe(self, df, table_name: str,schema_name: str = "DB",mode: str = "append",batch_size: int = 10000) -> Dict:
        """
        Write a DataFrame to an Azure SQL table.
        
        Args:
            df: DataFrame to write (either Pandas DataFrame or Spark DataFrame)
            table_name: Target table name
            schema_name: Schema name (default: "DB")
            mode: Write mode (append, overwrite, etc.)
            batch_size: Number of records to insert in each batch
            
        Returns:
            Dictionary with statistics about the operation
        """
        start_time = time.time()
        spark = self.get_spark()
        
        # Convert pandas DataFrame to Spark DataFrame if needed
        if isinstance(df, pd.DataFrame):
            if df.empty:
                logger.warning("No data to write to Azure SQL")
                return {"records_processed": 0, "records_written": 0, "elapsed_time": 0}
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = spark.createDataFrame(df)
            total_records = len(df)
        else:
            # Already a Spark DataFrame
            spark_df = df
            total_records = spark_df.count()
            
            if total_records == 0:
                logger.warning("No data to write to Azure SQL")
                return {"records_processed": 0, "records_written": 0, "elapsed_time": 0}
        
        # Qualify table name with schema
        qualified_table = f"{schema_name}.{table_name}"
        
        logger.info(f"Starting to write {total_records} records to {qualified_table}")
        
        # today = datetime.today()
        # spark_df = spark_df.withColumn("Created Date", lit(today))
        

        try:
            # Get connection properties
            jdbc_url, connection_properties = self.get_connection_properties()
            
            # Write the DataFrame to Azure SQL
            spark_df.write \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", qualified_table) \
                .option("user", connection_properties["user"]) \
                .option("password", connection_properties["password"]) \
                .option("driver", connection_properties["driver"]) \
                .option("batchsize", batch_size) \
                .option("createTableColumnTypes", self._get_column_types(spark_df)) \
                .mode(mode) \
                .save()
            
            records_written = total_records
            logger.info(f"Successfully wrote {records_written} records to {qualified_table}")
            
        except Exception as e:
            logger.error(f"Error writing to Azure SQL: {str(e)}")
            records_written = 0
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed writing operation in {elapsed_time:.2f} seconds")
        
        return {
            "records_processed": total_records,
            "records_written": records_written,
            "elapsed_time": elapsed_time
        }

    def filter_dataframe(self,df,store_id=None,item_id=None):
        """
        Filters Dataframe by store and/or by item for model forecast.
        """
        filtered_df = df

        if store_id is not None:
            filtered_df = filtered_df.filter(filtered_df['StoreID'] == store_id) 
        if item_id is not None:
            filtered_df = filtered_df.filter(filtered_df['itemid'] == item_id)


        filtered_df = filtered_df.select('ds','y')
        filtered_df = filtered_df.toPandas()
        
        return filtered_df
    
    def _get_column_types(self, df: SparkDataFrame) -> str:
        """
        Generate SQL column type definitions for a Spark DataFrame.
        
        Args:
            df: Spark DataFrame
            
        Returns:
            String with column type definitions for SQL CREATE TABLE
        """
        # Map Spark types to SQL Server types
        type_mapping = {
            "string": "NVARCHAR(MAX)",
            "integer": "INT",
            "long": "BIGINT",
            "double": "FLOAT",
            "float": "REAL",
            "boolean": "BIT",
            "timestamp": "DATETIME2",
            "date": "DATE",
            "binary": "VARBINARY(MAX)"
        }
        
        # Generate column type definitions
        column_types = []
        for field in df.schema.fields:
            sql_type = type_mapping.get(field.dataType.simpleString().lower(), "NVARCHAR(MAX)")
            column_types.append(f"{field.name} {sql_type}")
        
        return ",".join(column_types)
    


