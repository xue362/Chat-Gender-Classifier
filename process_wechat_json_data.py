import os
import json
import csv

def process_wechat_json_data(root_dir='json', output_csv='output.csv'):
    """
    遍历指定目录下的微信聊天记录JSON文件，动态查找文件名，并根据remark是否为空来确定聊天对象，
    最终提取信息并保存到CSV。
    """
    gender_map = {0: '未知', 1: '男', 2: '女'}
    
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            csv_writer = csv.writer(f)
            # 写入表头
            csv_writer.writerow(['姓名', '性别', '消息内容'])

            print(f"开始处理根目录 '{root_dir}'...")

            if not os.path.exists(root_dir):
                print(f"错误：根目录 '{root_dir}' 不存在。请检查脚本是否和 'json' 文件夹在同一级目录。")
                return

            for entry in os.scandir(root_dir):
                if entry.is_dir() and entry.name.startswith('wxid_'):
                    wxid_folder_path = entry.path
                    print(f"  正在处理文件夹: {entry.name}")

                    # --- 1. 提取用户信息 (姓名和性别) ---
                    name = '未知'
                    gender = '未知'
                    users_json_path = os.path.join(wxid_folder_path, 'users.json')
                    
                    try:
                        if os.path.exists(users_json_path):
                            with open(users_json_path, 'r', encoding='utf-8') as user_file:
                                user_data = json.load(user_file)
                                
                                # <--- 核心修改：遍历用户，找到remark不为空的那个（对方） --->
                                for user_info in user_data.values():
                                    # 如果remark字段存在且不为空，则认为是对方
                                    if user_info.get('remark'):
                                        name = user_info['remark']  # 直接获取，因为我们已经确认它存在
                                        extra_buf = user_info.get('ExtraBuf', {})
                                        gender_code = extra_buf.get('性别[1男2女]', 0)
                                        gender = gender_map.get(gender_code, '未知')
                                        break  # 找到对方后就跳出循环
                    except Exception as e:
                        print(f"    - 处理 users.json 时发生错误: {e}")

                    # --- 2. 提取消息内容 ---
                    messages_json_path = None
                    try:
                        files_in_dir = os.listdir(wxid_folder_path)
                        for filename in files_in_dir:
                            if filename.endswith('.json') and filename != 'users.json':
                                messages_json_path = os.path.join(wxid_folder_path, filename)
                                break
                    except Exception as e:
                        print(f"    - 查找消息文件时出错: {e}")

                    if messages_json_path:
                        try:
                            with open(messages_json_path, 'r', encoding='utf-8') as msg_file:
                                messages_data = json.load(msg_file)
                                for message in messages_data:
                                    if isinstance(message, dict) and message.get('is_sender') == 0:
                                        msg_content = message.get('msg', '')
                                        csv_writer.writerow([name, gender, msg_content])
                        except Exception as e:
                            print(f"    - 处理消息文件 {messages_json_path} 时发生错误: {e}")
                    else:
                        print(f"    -> 未在此文件夹中找到消息JSON文件。")

            print(f"\n处理完成！数据已保存到 '{output_csv}' 文件中。")

    except Exception as e:
        print(f"发生致命错误: {e}")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    process_wechat_json_data()