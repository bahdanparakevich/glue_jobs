import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import current_date, date_add
from awsglue.dynamicframe import DynamicFrame
from datetime import date

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# read csv file
dyf = glueContext.create_dynamic_frame.from_options(
    format_options={
        "quoteChar": '"',
        "escaper": '"',
        "withHeader": True,
        "separator": ",",
        "multiline": True,
        "optimizePerformance": False,
    },
    connection_type="s3",
    format="csv",
    connection_options={
        "paths": ["s3://deproject-bahdan/data/raw_data/csv/dataengineer/"],
        "recurse": True,
    },
    transformation_ctx="read_csv",
)

#add current date to new column "source_date" and add dates for partitionKeys
def add_date(r):
    r["source_date"] = date.today()
    r["year"] = date.today().year
    r["month"] = date.today().month
    r["date"] = date.today()
    return r

dyf = Map.apply(
    frame = dyf, f = add_date,
    transformation_ctx="add_date"
)

# rename and mapping columns
dyf = ApplyMapping.apply(
    frame=dyf,
    mappings=[
        ("source_date", "date", "source_date", "date"),
        ("Job Title", "string", "job_title", "string"),
        ("Salary Estimate", "string", "salary_estimate", "string"),
        ("Job Description", "string", "job_description", "string"),
        ("Rating", "string", "rating", "string"),
        ("Company Name", "string", "company_name", "string"),
        ("Location", "string", "location", "string"),
        ("Headquarters", "string", "headquarters", "string"),
        ("Size", "string", "size", "string"),
        ("Founded", "string", "founded", "string"),
        ("Type of ownership", "string", "type_of_ownership", "string"),
        ("Industry", "string", "industry", "string"),
        ("Sector", "string", "sector", "string"),
        ("Revenue", "string", "revenue", "string"),
        ("Competitors", "string", "competitors", "string"),
        ("Easy Apply", "string", "easy_apply", "string"),
        ("year", "int", "year", "int"),
        ("month", "int", "month", "int"),
        ("date", "date", "date", "date")
    ],
    transformation_ctx="mappings",
)


# save file in parquet format
output_parquet = glueContext.getSink(
    path="s3://deproject-bahdan/data/raw_data_parquet/dataengineer/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year", "month", "date"],
    compression="snappy",
    enableUpdateCatalog=True,
    transformation_ctx="output_parquet",
)
output_parquet.setCatalogInfo(
    catalogDatabase="deproject-glue-database", catalogTableName="dataengineer_raw_parquet_with_date"
)
output_parquet.setFormat("glueparquet")
output_parquet.writeFrame(dyf)
job.commit()
