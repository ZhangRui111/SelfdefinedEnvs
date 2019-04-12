# gui_maze

There are three objects except the player (yellow oval):

- exit (green oval)
  
  positive reward (default) & terminal
  
  **The player can reach the exit.**
  
- obstacle (gray rectangle)

  negative reward (default) & no terminal
  
  **The player cannot reach the obstacle.**
  
- void unit

  negative reward (default) & no terminal
  
  **The player can reach the void unit.**