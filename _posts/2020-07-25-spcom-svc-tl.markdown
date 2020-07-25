---
layout: post
title:  "Effectiveness of Transfer Lerning for Singing Voice Conversion"
date:   2020-07-25 00:00:00 +0300
image:  spcom-2020-svc-tl/svc-title.png
tags:   transfer-learning voice-conversion source-separation
---

Singing voice conversion (SVC) is a task of converting the perception of the source speaker's identity to the target speaker without changing lyrics and rhythm. Recent approaches in traditional voice conversion involve the use of the generative models, such as Variational Autoencoders (VAE), and Generative Adversarial Networks (GANs). However, in the case of SVC, GANs are not explored much. The only system that has been proposed in the literature uses traditional GAN on the parallel data. The parallel data collection for real scenarios (with the same background music) is not feasible. Moreover, in the presence of background music, SVC is one of the most challenging tasks as it involves the source separation of vocals from the inputs, which will have some noise. Therefore, in this paper, we propose transfer learning, and fine-tuning-based Cycle consistent GAN (CycleGAN) model for non-parallel SVC, where music source separation is done using Deep Attractor Network (DANet). We designed seven different possible systems to identify the best possible combination of transfer learning and fine-tuning. Here, we use a more challenging database, MUSDB18, as our primary dataset, and we also use the NUS-48E database to pre-train CycleGAN. We perform extensive analysis via objective and subjective measures and report that with a $4.14$ MOS score out of $5$ for naturalness, the CycleGAN model pre-trained on NUS-48E corpus performs the best compared to the other systems described in the paper.


<body>
    <div>
        <table>
            <tbody>
            <tr>
                <td>
                    <audio controls src="/audios/spcom-2020-svc-tl/results/00001.wav" type="audio/x-wav;"/></audio>
                </td>
                <td >
                    <audio controls src="/audios/spcom-2020-svc-tl/results/00002.wav" type="audio/x-wav;"/></audio>
                </td>
            </tr>
            </tbody>
        </table>
    </div>
</body>

<!-- <audio controls>
 <source src="/audios/spcom-2020-svc-tl/results/00001.wav" type="audio/x-wav;"/>
 <source src="/audios/spcom-2020-svc-tl/results/00002.wav" type="audio/x-wav;"/>
</audio>
</body> -->

<!---
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
--->