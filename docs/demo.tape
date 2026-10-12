# Re-record the README demo: `vhs docs/demo.tape` from the repo root.
# Needs data/binance-top100.npz (`falsealpha data binance --out data/binance-top100.npz`).
Output docs/demo.gif

Set Shell zsh
Set FontSize 13
Set Width 1240
Set Height 820
Set Padding 12
Set Framerate 15
Set TypingSpeed 45ms
Set Theme { "background": "#0d0d0d", "foreground": "#e0e0e0" }

Hide
Type "export PATH=$PWD/.venv/bin:$PATH && rm -rf /tmp/falsealpha-demo && clear"
Enter
Show

Type "falsealpha info data/binance-top100.npz"
Sleep 300ms
Enter
Sleep 4s

Type "falsealpha mine data/binance-top100.npz /tmp/falsealpha-demo --generations 10"
Sleep 300ms
Enter
Sleep 26s
