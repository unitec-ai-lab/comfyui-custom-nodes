from zai import ZhipuAiClient
import requests

url = "https://open.bigmodel.cn/api/paas/v4/files"

files = { "file": "open('example-file', 'rb')" }
payload = { "purpose": "batch" }
headers = {"Authorization": "Bearer 73c95dbd36ed48bdbe79ead423219d4e.8pZApezlT8JLM4Rr"}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.json())


client = ZhipuAiClient(api_key="59856fbdedc949b993a04e2c057aa6fd.y1mbUzy6KYfhx5lK")  # 填写您自己的APIKey

# 创建 Batch 任务
create = client.batches.create(
    input_file_id="file_123",
    endpoint="/v4/chat/completions",
    auto_delete_input_file=True,
    metadata={
        "description": "Sentiment classification"
    }
)
print(create)

# 检索 Batch 任务
retrieve = client.batches.retrieve("batch_123")
print(retrieve)

# 取消 Batch 任务
cancel = client.batches.cancel("batch_123")
print(cancel)

# 列出 Batch 任务
# after	String	非必填	此参数用作分页游标，指定从特定对象ID之后开始检索列表。例如，如果您的上一请求返回了包含对象 obj_foo 的列表，并希望继续从这一点获取后续内容，可以将 after=obj_foo 包括在您的下一请求中以获取下一页数据。
# limit	int	非必填	限制返回对象的数量。limit 的范围可以是 1 到 100，默认值为 20。
batch_list = client.batches.list(limit=10)
print(batch_list)
# SyncCursorPage的get_next_page 可用于获取当前 after+1的数据
next_page = batch_list.get_next_page()
print(next_page)
# SyncCursorPage的iter_pages 返回一个分页迭代器，可以使用collections相关api
for batch in batch_list.iter_pages():
    print(batch)

# 下载 Batch 结果
# file_id	String	必填	被请求的文件的唯一标识符，用于指定要获取内容的特定文件。
# client.files.content返回 _legacy_response.HttpxBinaryResponseContent实例
content = client.files.content("result_123")
# 使用write_to_file方法把返回结果写入文件
content.write_to_file("write_to_file_batchoutput.jsonl")