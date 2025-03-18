# STAGE - UQ for Material Science

**Zacharie FREDJ-ZADOUN**, CentraleSupélec (Saclay)

Encadrants :

- **Inna KUCHER** (DES/ISAS/DM2S/SGLS/LIAD)
- **Riccardo FINOTELLO** (DES/ISAS/DM2S/SGLS/LIAD)

Ce repo git sert à regrouper notes et code du stage. Il ne faut pas hésiter à modifier la structure ou à améliorer la gestion.

> **⚠️ Attention !**
> Code et notes doivent être compréhensibles et réutilisables !

Voici quelques suggestions pour le déroulement du stage. Il est fondamental de garder un esprit critique : toute proposition est discutable et passible d'être modifiée ! 😄

Conseil d'utilisation :

1. créer une branche `rapport` pour le rapport de stage en $\LaTeX$ ($\neq$ code du stage) : `git checkout --orphan rapport` ;
2. créer une branche `pres` pour le support de présentation pour la soutenance : `git checkout --orphan pres` ;
3. créer une branche `dev` pour le développement du code lié au sujet de stage (ou plusieurs branches selon le type de projet) : `git checkout -b dev` (**à partir de la branche ``main``**);
4. créer une branche `dev_rapport` pour le développement du rapport : `git checkout -b dev_rapport` (**à partir de la branche ``rapport``**) ;
5. créer une branche `dev_pres` pour le développement du support de présentation : `git checkout -b dev_pres` (**à partir de la branche ``pres``**).

Pour le déroulement du stage :

- utiliser les rapports ("issues") pour discuter des différents sujets et assurer le suivi des avancées ;
- créer des branches à partir des "issues" (mais pas que…) pour implémenter des caractéristiques différentes ;
- commenter et tester le code (on peut discuter de CI/CD, au cas où) !
