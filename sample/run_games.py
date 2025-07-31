import subprocess
import time
import re
import sys
import os
import csv  ### 1. csvモジュールをインポート

def run_single_game(game_number, log_threshold=80):
    """
    1回のゲームを実行し、結果を返す
    """
    print(f"--- Starting Game #{game_number} ---")
    
    server_cmd = [sys.executable, "server.py", "--games", "1"]
    random_player_cmd = [sys.executable, "random_player.py", "localhost", "2000"]
    original_player_cmd = [sys.executable, "original_player.py", "localhost", "2000"]

    server_proc = None
    random_player_proc = None
    original_player_proc = None

    try:
        server_proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Server started...")
        time.sleep(0.025)

        random_player_proc = subprocess.Popen(random_player_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("RandomPlayer started...")

        original_player_proc = subprocess.Popen(
            original_player_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        print("OriginalPlayer started, waiting for game to finish...")

        stdout, stderr = original_player_proc.communicate(timeout=60)

        if "you win" in stdout:
            result = "win"
        elif "you lose" in stdout:
            result = "lose"
        else:
            print("Game did not finish correctly. Error log:")
            print(stderr)
            result = "error"

        turns = re.findall(r"t=(\d+)", stdout)
        last_turn = int(turns[-1]) if turns else 0

        ### 2. ログ保存のロジックを追加 ###
        if last_turn > log_threshold:
            # ログを保存するフォルダを作成（なければ）
            log_dir = "long_games_log"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # ファイルにログを書き込む
            log_filename = os.path.join(log_dir, f"{game_number}_log.txt")
            with open(log_filename, 'w', encoding='utf-8') as log_file:
                log_file.write(f"--- Game #{game_number} Log ---\n")
                log_file.write(f"Result: {result}, Turns: {last_turn}\n")
                log_file.write("-" * 20 + "\n")
                log_file.write(stdout) # プレイヤーの標準出力をすべて書き込む
            print(f"Game took {last_turn} turns (> {log_threshold}). Log saved to {log_filename}")

        print(f"Game #{game_number} finished. Result: {result}, Turns: {last_turn}")
        return result, last_turn

    except subprocess.TimeoutExpired:
        print(f"Game #{game_number} timed out!")
        return "timeout", 0
    except Exception as e:
        print(f"An error occurred during game #{game_number}: {e}")
        return "error", 0
    finally:
        print("Cleaning up processes...")
        if original_player_proc: original_player_proc.kill()
        if random_player_proc: random_player_proc.kill()
        if server_proc: server_proc.kill()
        print("-" * 20 + "\n")


def main():
    """
    指定回数のゲームを実行し、統計情報を表示・CSVに出力する
    """
    num_games = 100
    results = []
    turn_counts = []
    csv_filename = "game_stats.csv"  ### 2. CSVファイル名を定義

    ### 3. CSVファイルを開き、書き込み準備 ###
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # ヘッダーを書き込む
        csv_writer.writerow(['Game_Number', 'Result', 'Turns'])
        
        # メインループ
        for i in range(1, num_games + 1):
            result, turns = run_single_game(i)
            results.append(result)
            if result == "win":
                turn_counts.append(turns)
            
            ### 4. 1行ずつ結果をCSVに書き込む ###
            csv_writer.writerow([i, result, turns])

    # 統計の計算
    wins = results.count("win")
    losses = results.count("lose")
    errors = results.count("error") + results.count("timeout")
    
    win_rate = (wins / num_games) * 100 if num_games > 0 else 0
    avg_turns_on_win = sum(turn_counts) / len(turn_counts) if len(turn_counts) > 0 else 0

    # 最終結果の表示
    print("\n" + "="*30)
    print("      FINAL RESULTS")
    print("="*30)
    print(f"Total Games Played: {num_games}")
    print(f"Wins:     {wins}")
    print(f"Losses:   {losses}")
    print(f"Errors/Timeouts: {errors}")
    print(f"Detailed results saved to: {csv_filename}")
    print("---")
    print(f"Win Rate: {win_rate:.2f}%")
    if avg_turns_on_win > 0:
        print(f"Avg. Turns on Win: {avg_turns_on_win:.2f}")
    print("="*30)


if __name__ == "__main__":
    main()