from small_sem import Game, Player

p1 = Player(_player_type="Human")
p2 = Player(_name="0200k_test_sem3_3x4_dqn_16", _player_type="DQN")

game = Game(p1, p2)

game.play_gui()