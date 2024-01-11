import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import when, col, regexp_extract, coalesce, split
from awsglue.dynamicframe import DynamicFrame
  
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

dyf = glueContext.create_dynamic_frame.from_options(
    format_options={},
    connection_type="s3",
    format="parquet",
    connection_options={
        "paths": ["s3://deproject-bahdan/data/raw_data_parquet/dataengineer/"],
        "recurse": True,
    },
    transformation_ctx="dyf",
)

df = dyf.toDF()

columns_renaming = {
    "location": "job_location",
    "size": "company_size",
    "industry": "company_industry",
    "sector": "company_sector",
    "revenue": "company_revenue",
}

for old_col, new_col in columns_renaming.items():
    df = df.withColumnRenamed(old_col, new_col)

#drop rating from company_name
df = df.withColumn("company_name", split(col("company_name"), r'\s+\d+\.*\d*')[0])

patterns_for_experience = [
    r'(\d+-\d+|\d+\+|\d+) years of professional experience',
    r'(\d+-\d+|\d+\+|\d+) years professional work experience',
    r'(\d+-\d+|\d+\+|\d+) years of Data Engineering experience',
    r'(\d+-\d+|\d+\+|\d+) years of professional software development experience',
    r'(\d+-\d+|\d+\+|\d+) years of work experience',
    r'(\d+-\d+|\d+\+|\d+) years of experience in Data Engineering',
    r'(\d+-\d+|\d+\+|\d+) years experience',
    r'(\d+-\d+|\d+\+|\d+) years of experience',
    r'(\d+-\d+|\d+\+|\d+) or more years experience',
    r'(\d+-\d+|\d+\+|\d+) years’ experience'
]
pattern_for_min_experience = r'(\d+)'

for i, pattern in enumerate(patterns_for_experience):
    df = df.withColumn(f"temp_{i}", regexp_extract("job_description", pattern, 1))

df = df.withColumn("job_experience",
                   when(col("temp_0") != "", col("temp_0"))
                   .when(col("temp_1") != "", col("temp_1"))
                   .when(col("temp_2") != "", col("temp_2"))
                   .when(col("temp_3") != "", col("temp_3"))
                   .when(col("temp_4") != "", col("temp_4"))
                   .when(col("temp_5") != "", col("temp_5"))
                   .when(col("temp_6") != "", col("temp_6"))
                   .when(col("temp_7") != "", col("temp_7"))
                   .when(col("temp_8") != "", col("temp_8"))
                   .otherwise(col("temp_9"))
                   )

df = df.withColumn("job_min_experience", regexp_extract("job_experience", pattern_for_min_experience, 1))
# reformat from String to Integer
df = df.withColumn("job_min_experience", col("job_min_experience").cast("int"))
jbl_pattern = r'((?i)senior|(?i)sr\.|(?i)lead|(?i)junior|(?i)intern)'

df = df.withColumn("job_level", regexp_extract(col("job_title"), jbl_pattern, 1))

# Get job level from description based on experience
# if exp is less than 3 put "regular" otherwise "Senior"
df = df.withColumn("job_min_snr",
                   when(col("job_min_experience") >= 3, "Senior").otherwise(
                       "Regular"))
                       
# combine the results of getting lvl from tittle and experience
# if info from tittle is NULL get value from experience
df = df.withColumn("job_level", when(col("job_level") == "", col("job_min_snr")).otherwise(
    col("job_level")))

# Get Salary Estimate and split it to min and max
pattern_for_salary = r'\$(\d+)K-\$(\d+)K'
df = df.withColumn("min_salary_usd",
                   regexp_extract(col("salary_estimate"), pattern_for_salary, 1).cast("int") * 1000)
df = df.withColumn("max_salary_usd",
                   regexp_extract(col("salary_estimate"), pattern_for_salary, 2).cast("int") * 1000)
df = df.withColumn("avg_salary_usd", ((col("min_salary_usd") + col("max_salary_usd")) / 2).cast("int"))

keyword_tools = [
    ("(?i)python", "python"),
    ("\\b(?)java\\b", "java"),
    ("\\b(?)scala\\b", "scala"),
    ("(?<!l)aws|amazon web services", "aws"),
    ("azure", "microsoft_azure"),
    ("gcp|google cloud", "google_cloud")
]

language_cloud_columns = [when(col("job_description").rlike(f"(?i){keyword}"), True).otherwise(False).alias(column_name)
                 for keyword, column_name in keyword_tools]

df_final = df.select(
    "source_date",
    "job_title",
    "company_name",
    "job_location",
    "company_size",
    "company_industry",
    "company_sector",
    "company_revenue",
    "job_experience",
    "job_min_experience",
    "job_level",
    "min_salary_usd",
    "max_salary_usd",
    "avg_salary_usd",
    *language_cloud_columns)


dyf = DynamicFrame.fromDF(df_final, glueContext, "dyf")

s3output = glueContext.getSink(
  path="s3://deproject-bahdan/data/output_data_parquet/dataengineer/",
  connection_type="s3",
  updateBehavior="UPDATE_IN_DATABASE",
  partitionKeys=[],
  compression="snappy",
  enableUpdateCatalog=True,
  transformation_ctx="s3output",
)
s3output.setCatalogInfo(
  catalogDatabase="deproject-glue-database", catalogTableName="DataEngineer_trasformed"
)
s3output.setFormat("glueparquet")
s3output.writeFrame(dyf)
job.commit()
