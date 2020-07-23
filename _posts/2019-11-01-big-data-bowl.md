---
title: "2019 NFL Big Data Bowl"
date: 2019-11-01
tags: [machine_learing, kaggle]
header:
  image: "https://nflops.blob.core.windows.net/cachenflops-lb/6/f/e/b/f/a/6febfa757c85f993b8aabc450d4aa0e5452ca938.jpg"
mathjax: "true"
---

During the 2019 football season, the NFL and Kaggle teamed up to launch an awesome [competition](https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview/description).  The goal: predict the number of yards a running back will gain at the time they are given the ball.  In order to do this, contestants were given player traking data, which held player coordinates on the field for each position, along with some other detailed information such as player speed, player position, stadium type, field type, etc.

After weeks and weeks of working on the competition, my model used a blend several techniques from clustering to using Voronoi diagrams to a Neural Network for prediction.  Ultimately, I was able to place 391st out of 2,038 teams **(TOP 20%)**.

<img src="{{ site.url }}{{ site.baseurl }}/images/BDB_leaderboard.png" alt="Big Data Bowl Leaderboard">

For a look at my code, checkout my [Notebook on Kaggle](https://www.kaggle.com/mtodisco10/final-big-data-bowl-submission).
