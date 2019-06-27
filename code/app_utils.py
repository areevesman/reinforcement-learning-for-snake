import pygame

from agent import *

class App():
    def __init__(self, agent=Snake(), caption='Q-Learning - Score: 0'):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.agent = agent
        self.caption = caption
        self.Q = {}
        states = []
        false_true = [0,1]
        for a in false_true:
            for b in false_true:
                for c in false_true:
                    for d in false_true:
                        for e in false_true:
                            for f in false_true:
                                states.append((a,b,c,d,e,f))

        actions = [i for i in range(3)]
        state_actions = [(s, a) for s in states for a in actions]
        for sa in state_actions:
            self.Q[str(sa)] = 0

    def on_init(self):
        """start app and load images"""
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.agent.game.window_width,
                                                      self.agent.game.window_height),
                                                      pygame.HWSURFACE)

        pygame.display.set_caption(self.caption)
        self._running = True
        self._head_surf = pygame.image.load("../images/snake_head.png").convert()
        self._image_surf = pygame.image.load("../images/snake2.png").convert()
        try:
            self._apple_surf = pygame.image.load("../images/jeff.png").convert()
        except:
            self._apple_surf = pygame.image.load("../images/apple.png").convert()

    def on_render(self):
        """show board in app"""
        self._display_surf.fill((0, 0, 0))
        self.agent.draw(self._display_surf, self._image_surf,self._head_surf)
        self.agent.game.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        """quit app"""
        pygame.quit()