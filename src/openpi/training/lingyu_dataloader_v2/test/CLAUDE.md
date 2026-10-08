# test/ 目录代码规范

## Logger
- 所有 test 文件必须从 `openpi.training.lingyu_dataloader_v2.test.logger` 导入 `logger`，禁止在 test 文件内单独配置 `logging.basicConfig`

## 文件命名
- 测试文件统一命名为 `test_<被测模块名>.py`

## 文件夹结构
- 与 lingyu_dataloader_v2 中的文件结构相同

## 结构
- 使用 pytest test/ -s 来 在终端中实时打印 print() 的内容（关闭 pytest 的输出捕获）
