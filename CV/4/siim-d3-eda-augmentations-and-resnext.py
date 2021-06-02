#!/usr/bin/env python
# coding: utf-8

# <div align="center">
# <font size="6"> SIIM: d3 EDA, Augmentations and ResNeXt</font>  
# </div>
# <br>
# <div align="center">
# <font size="4"> Build tools for melanoma detection </font>  
# </div>
# 
# ---
# 
# **Trigram**
# 
# ![](https://siim.org/resource/resmgr/home/jdi_banner-2000x550.jpg)
# 
# ---
# <b><a href="#one">1. Introduction</a><br></b>
# <b><a href="#xxx">2. A neat trick wth 3-dimensional visualization</a><br></b>
# &emsp;&emsp;<a href="#exp">2.1. Explanation of our method</a><br>
# &emsp;&emsp;<a href="#app">2.2. Application of our method</a><br>
# <b><a href="#two">3. Benign and malignant tumors</a><br></b>
# &emsp;&emsp;<a href="#benign">3.1. Benign image viewing</a><br>
# &emsp;&emsp;<a href="#malignant">3.2. Malignant image viewing</a><br>
# <b><a href="#three">4. Which part of the body?</a><br></b>
# <b><a href="#four">5. Diagnosis</a><br></b>
# <b><a href="#five">6. Age</a><br></b>
# <b><a href="#ca">7. Image preprocessing</a></b><br>
# &emsp;&emsp;<a href="#norm">7.1. Normalize images</a><br>
# <b><a href="#six">8. Image Augmentations</a><br></b>
# &emsp;&emsp;<a href="#gray">8.1. Grayscale images</a><br>
# &emsp;&emsp;<a href="#gray">8.2. Ben Graham's method from first competition</a><br>
# &emsp;&emsp;&emsp;<a href="#graytrain">8.1.1. Ben Graham's method on training set</a><br>
# &emsp;&emsp;&emsp;<a href="#graytest">8.1.2. Ben Graham's method on testing set</a><br>
# &emsp;&emsp;<a href="#neuron">8.3. Neuron Engineer's method from APTOS</a><br>
# &emsp;&emsp;<a href="#circ">8.4. Circle crop</a><br>
# &emsp;&emsp;<a href="#autocrop">8.5. Auto cropping</a><br>
# &emsp;&emsp;<a href="#bgsub">8.6. Background subtraction</a><br>
# &emsp;&emsp;<a href="#seg">8.7. Image segmentation</a><br>
# &emsp;&emsp;<a href="#segf">8.8. A finer form of image segmentation</a><br>
# &emsp;&emsp;<a href="#segfg">8.9. A finer form of image segmentation: Grayscale form</a><br>
# &emsp;&emsp;<a href="#fourier">8.10. Fourier method for image pixel distributions</a><br>
# &emsp;&emsp;<a href="#albumentation">8.11. Albumentations library demonstration</a><br>
# &emsp;&emsp;<a href="#erosion">8.12. Erosion</a><br>
# &emsp;&emsp;<a href="#dilation">8.13. Dilation</a><br>
# &emsp;&emsp;<a href="#combination">8.14. Combination of erosion and dilation</a><br>
# &emsp;&emsp;<a href="#microscope">8.15. Roman's microscope augmentation</a><br>
# &emsp;&emsp;<a href="#put">8.16. Albumentations + erosion</a><br>
# &emsp;&emsp;<a href="#put2">8.17. Albumentations + dilation</a><br>
# &emsp;&emsp;<a href="#put3">8.18. Albumentations + dilation and erosion</a><br>
# &emsp;&emsp;<a href="#wl">8.19. Complex Wavelet Transform</a><br>
# 
# <b><a href="#et">9. 3-dimensional image augmentations</a><br></b>
# <b><a href="#seven"> 10. Baseline modeling</a><br></b>
# &emsp;&emsp;<a href="#3dgray">10.1. 3-dimensional grayscale images</a><br>
# &emsp;&emsp;<a href="#grassroots">10.1. Grassroots (bulding from scratch)</a><br>
# &emsp;&emsp;<a href="#qubvel">10.2. qubvel's method</a><br>
# <b><a href="#eight">11. CutMix, MixUp and GridMask</a><br></b>
# <b><a href="#nine">12. Even more data</a><br></b>
# <b><a href="#ten">13. Attack the model with adversarial learning</a><br></b>
# 
# 
# <a href="#app">Appendix</a><br>
# &emsp;<a href="#a">A. About Melanoma</a><br>
# &emsp;<a href="#b">B. About SIIM-ISIC organization</a><br>
# &emsp;<a href="#c">C. Past Winner's Solutions</a><br>
# 
# ---
# 
# 
# <img align="right" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQkAAAC+CAMAAAARDgovAAAAkFBMVEX///8XNmAUNF8NMFwuR2wXOGJid5MvSW9DXH5KX3/4+foRM14mQmju8fRyg5xabouKmK2BkKXR1+CZpLVeco9CW37Iz9hofJfi5uva3+Xp7PAALFrx8/YxTnTAx9Lh5eqirr+4wMyvuMYAKFh6iqCVorNRaYmPnK+gq7s5VXkdP2nDytNSa4tsgJqqtcVod5GbDTtHAAATxUlEQVR4nO1dC3uyvLKlUECC3BFIRG6FcLHg//93JxFrUTGA9tvf2eewnud920oIw0oyk5lMIsetWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixf8TAAA0RKGFAPyrYoBeDLBMDBAyMaMuoMV2Wpmtu9sdetSKlDupB7V5krwvwqUeZEdZ6Su7uhdjp0hWldrBzBoyi4lo6uEwcnxFFgUKVVB7kN/3QrORyhRq0y+QMiUwj7PeA9lp+X0Q92c5foSgv8tYd45xOF1Du1dZMFn3hnaX4y1PHvcxAp4Qs8VWBieaRLMEpgjOdJOGdpZvGvK8D/5BDPKpeDiVEZqqxN9/8Hf4/eDjg8EESi1Mnv747Bs21MYoPWaLaJb6MfLsqwiTTIDIwltVZQiiCh+HtosnmWDhKROoO8nMp1/JEMRdHjHehjLBgDjBBIjyeqI9ejK2hsPkgs2E+IQJEPnsVriV4uNgPRfiPSagdeCZ91/Bq1sjZdT1Up8IrO1sGs5Q1V36TzAB0o06jwcKXm2s4E+ZsHf7+Y+/QOCfCfEGE4ElCoukUIXNU1O0fHSEmbzs8T14wbVH3+l1Juzv/aKeeZaiyZ7Ut7hPoEJe3CF6qMao4nyZiSNmyz4OQU7G5zhLmdBeJoI0CB6j4lUmIuU1QdRDNWrVFzIROp8vE0GoUEZG6YtMRHjpyPiBeujGqlzGBOhe7xEUwgb+EROe8ioRlIoxQ7ZMY0aHt4ggCuv7YZC+xISN3xFE2Nlv9ol484rVGILf5/cv9goTaELsKez9RzdkERP6u0QQKrbd+0wA572uScQoH2pdMjqixeZ7BOrGe5uJtHmXCbV5iDcs6BOh8X6XIBDzW1WxnAn49iAlqsK4N6ULmMi2f9AlSHPUt5p7MROAfcM88OL9KJ3PhHb6AwGoDGp+o64WM3Gs/0IQ1bhTmvOZeNeC/sog33SKCSb4eybCdpkf/Kzee9XdztWYoSVOVn6JMk1B8Idu6dI+0W3/pkVU99Y5nt0noMEWgFfF5vMgH7bNx1TIQN0OO8VCJpA+TQQNLQtTcSxVzl5jImUPDnXrmkmWpmlWWadaZEsr6APzsZCJCTlo7eLmpOutMRlNsm6M2Nz5BKgaVkFeSYIfiUEQlZvno4SGvIfDYxkTKJ94PV7FlYfCUIOZzp52qMrNnHtunwiZ8qqH2/hH6OnNqMSqwMtumQ4j/8uYmDIcPK9f3w9VNatbEJ05rHouE8GJNZtR3fuILSrFByHI2JWlyrtbhlnEBCgm9LaaDyQJsx2rB92a87lM2My4CJ8/eDTAuqmZF/ZfB7+zH5fmFjGBJvS2YNy4/ewogqoMC8/VEx6zV6qnxyg+8H96Ea+q/FYxvfGo2SImoseediOHfOdOAJPRh/gb52Nun/CYIRp+OxIRizf0FlUVP5U2eYzQvMLEXUd7lMO6ZxtuGLXvh3XPZeIos5f9DsXD+idIGkFtatdMmUujS5jQdmxrUD8Go1jTj/3Qms9lIppwv1TRcKK7lXEo7XQniidXiOczYY9bpB/w/uOaSscYHupmoN7m6onoc2oarYqyYTmpja6SA3vwx58wkTClJWP0sXrICEHz28Gg/Rs90YNYSaITTlYVzclXeIUJiSmtWnuP1YesYAY/GE2zmZjpCvOqIDYHpS3SYLo7LGUiZDeHaowtOEoMJvaDTvQ384k7ULPZ1JJznEzfWMaEzbahvD5Gfs54w731e8dcPcGeY46JRTOMtmRKOZVttYCJlB1IFUfUBMc5rD7R/g7j2X6H+Up8RBW+xB2xoiw2FjBRfLFK8o9RWoqO4Z4L+NfazWWC+KIvRoqI4tjkjGyr+UwAna0w5dHpW3rbJ3iep8GLPodrX/+O39nxieiN6OE5tWc8Z2ARE5g5QtV6VC1F+3Ma3p6CGLemkeVDvVOwdNJzy5k7OgbRu+CtgC6vNrgYn3HPZyKcmGHiUdvtyYeN4kq6ZTpORYNJR8+2IYRBgDTthdk2B5xXh0cPXhVxNibqfCZidrhKMEaZBuSVz5nDIWBl7YL5se1FdnQMvPqZjxj8+UxMzO4E6elrzsF8JoD1dnBd3eN3sgYmfJ99/g8ycbMuCuu3F+H4vfJAxXwmMmYolUyT/kEmbtfKnfeW6ntxlXvfYD4TLLeSVs1Mrf5TJjTlD7IGVHyXxTGfiYRd9VfxH2OCi97LLbq8mn6rNv87mLjLLgLVxJLOHPBNcWNM5zPhsFX2f7JPcGE5M0maBfVw4x/MZ6Jg1/seEwvmE73Y5tvpLNTwDyPh/519gshdvpOQ2YO/jdLO1xP/IhMjedtaghekz49D2A1mFQuYYFvRr/IfZGI0l9/Ot9M7S9gQkleY+N8zn7jKnknNe1wI+Hd4/DfOMX8RJ4b4Dhe8+Du9+ju/Q3+PCea08dnuJw6gzOW/Xp9yDrTbfCZs9sRu77/HxIs74ggZ0KxF4UXtKXy/wETADps9iU/MxNL5xN3dnokPIv9CrFeVr6HUP4tZDZXPEKUitZZZdZdgFULnkM1D0OYtJghQVJ522w9BnZN0N3zBq6JYEMf8nohjju40A8aXcIHYyIcaY9fXc7NMsjQ6DoKrr+2SvCXD60zChrof30s8Cn5/TXxbEOWfiG1/juxVuC4HnhuKv26zpns8RbF5Ico/jhD1S8Bh4KWFjun+8plD5XcdbgETFXu9Yzu63oFG8zd7KQW8fOXnHiCMvcyxTvrVhQAIelnp181kJuRZhuuEcAETEdtYic6YpB5rDUyfOzpGmQAhTBPLx/Vn8yGIN7kbgNBxNHEzPU6E6zRoARMxM7zO8w+7aCgKFhPly3pCs9OiVeqt+KHSkcZ/PO7hIYU8azs11RCu06AFTICJ7NjRtXLWG+67eeU+fvsEaWz7WJGur3597Yc7ynlxdFMubCeyBV9igrnu/RD46BEzpmN8MwiqzmECRp31fWj2X/sRDSDc7w7ogdqJdMFXRsfkYrn5uLRUMcrf2N0ZTACdHvXx1Cao5uiEZiJfULiOuyVMsBqYinIfLSa1u4w7BGl2xlmvJ0z2hGabjFERTDgJr1jRyQ1YTXnfKTpWfthQYc4aHd5EOujo9mTIdBx54WpzFuVjTiy53OePc7bBStxuhoZvDhPhjt0S6jb3HroFO+jIvzLb5ibd0Q91c6M0oc9S3Dc7moE058Qea2J7pPpRW7dcIOfAHlI3Hhir3W7PqUHt1ME49WCsehJrVYIX9KGyt+oNC/1otrfNdgKN7FYRPHt5IYxKPFFc/PWgNZNdeXVDcSZPCdK0l0aJnZpds3yj4ArDZaGfeQCz1afR5pZZFkVpWvlkcel35xHI2EVvz5YJymlR8rJKksqxpsqZN2n3/tVlHcXF1qWNOAd8P6Sol8fGh/g7QjWLVf5+pwtIPqdr793NCTHuZx/z5phIEh5OmmLhPAwZl4c+4M15ViM13e0Dgy6j+MfEg29kuEt3n+d3gOy91KJ7qMmvBAv3i4Lkb7Yyf/B35w3MGx1cIO2Z5ZZhf7vyw666uDPQgfE3onzebpLkyjkakyDFzHLLgIdKO0yYZaWHs4ZStsjzIBn3qc2zrChtuZJZbhF2N0dbhcWOicdDZZzdH8iA73N7/LlnIXrK8MhC9vGFbAi3m3eXn4UYbNhCz8Fjetr8fMxkcq/LTE0lmjdbg5busCdI387tEZSHoI6/Z1galR/ErEJzkSF9jrt9hKRPMEUYO50l2bIs7zSEkdOL8q3MwjAlITipzLLz0Nwn34Um+4ZqxOcPyQz9DRm2Y5UWrsTCzYiGzKLz4Lb355yFCfuO0XPqApMt94QQzkicTd9+snCbpnLcsUtPY3u/tZV6YOxKx/oEca/yN4TQxyO/zAF1f7JXvVffGJ7qvnkcn9QrZ+HJaW9IF14TRRXa0WNLF6532MbC9c8hePExzjgVnxi1HWeA15Lf1OZhn/EPEyz67vsEoaJtXu0V6qc+tsODzidYIjw9FRIU8nJR1G355OBra6Ow8LjHLDbrmnnLM9RKNboeEDrs+0YPLuypyPBSUQ7KaPiZosQGC/eHPlHJo5J5yxNIzmOss6+vYt/47IxTCrtqFwnhFyN7ay9oBUYo48kKcVDsVH4qXHIbOhHd9NneUTrHZMVTmGcra6krsm4fgudxxtjA+tJaObCtw/xkCV4QlerlU6af64leFFjsZomiCgfz2aa815mg579b8tccCYjlFL4z1s7qV23HlYvYbKZE4YUvVWfysMAXfQDqXJpj9fw1aFRT3CrlxMH8xHYwndEZ5/KjxJXp4v3Isfw/J6I82Zg4wER8YjQ344q404162wdy+Z+JBjVV6pmEA9Yd9pn8FGHBjiQ8tx1DQPqVEc1Qkosc4nYnlccZZx8ENhPPjyzvAZDXlflJqQ/ytjkHwBviGx1qxdXNJJr6moZZIszZpU8RwrTIT3h3+GyoJFSOWjF0M7OnvznjjwDC2D6mWVJVRVkWFd2d6tlzv8bkjyUJbC/NqsopCrpLNvLgjKNA/gE5fvAvPHtUkH9bjhUr/o8A2SCwQ/LD9jyi5uh/iPwfAwCJ4ocBmcPZxD4QQKLEgZ2S6QkilhGRq5qnkcu9dQnPVziY0gNZ+mJ2ENJKe5CSRIWfP0D95YA+iqOTRITIL+H54V58LgEh/TymNZ0f6tGHEpBC5PO+IHd+mkbzpclE/iwueQKKIhTS6/SFyEcBfQt6E6kD9q9BPowiIvW5rl+z0m3saptxFc4VWcnbrWvkmYR9pQukjQ2kioslJT66texauOOqjYsrkKkVl1qIq8SKC6SzbxdWWKo7zjYMLCHQKYbSabvK3mxc1zfknUGupUbARRvFkNIOu7uKK2TfaMkbB99JdrCAjUvjsDOKA7klanV6/KNJCK8TQIpvCtDJ2PXtg4Ti3PKJOMQv84g0lgapD2ajkwK57yjWiasXSRsZm987HZuocjXOysOMPLKwsKzo1kGLW2xIkNsYMWp/EweTnV01uua4QaZkgVXHMbKNCpXY9ptKcx3Ow3UUxpZhezhBcoEqDLutgdIchT7WucA9B9egW6DOCc0WHus02JhBIcHasYm7EQcQW9DyY59M0qJdFsf2d4WKmj7iqBBXN8BVVtfQxp59KqF3SIJY81sak6JMHBJoOEGKgwRHQRDIShq3UWBh0pjAl1DkRhW2Y2zCtkk4nEb4GJvHuMJR7OooU46OQZmArhl0tRd9d8iRUWXYtlGCXd3FfnTDhOF6jsQRUTizjhGARgcKbOtuC0mf6HS/AFx5CqGR2FuPNMQx+faz1NJsXGF0YSLwfQ+FgVFxoZTbdcrFaXBmIiASfxeETwuT/hxtUg1EG1LLwS6M2MYXJiSjsL/jMO84eEiQBlr/l4mj4nHACwkTCKBdbnmtzTkSoMfqJVyY2pYOuPzk5ZIf4tQjs2oyg0gNCE46iH6YOCopp3UalI5cIiM/ByCXQqznxzsmLNMsJSIsYWLb5jY0jFxJgtbRM/KaVldJYc9EF31CQlia+GnrWFrmB/h4YQJE7ka3YyXjOF2KdjQIEBImNthPAGUCWA3N0452rt51mNRy8ApZl3w6OnCStQlOjTjUEw42uO209rdPVCmGsEpQIru6hxTyZMqEG9KjD2hMFOkWaSj3mHd+aqSaszEcrWdiZxkW6plI6coCURguZSJwTcCZCjKSvLhjovR8ye+ZqInigUae1wi1aSYpCcJWXkPOPDNx3NqcraSJpEmGFeRGVzuxdAk9o1QxPNrGun+sj/Qb8OrKVhyIzn2Cy2T7zER1hJlin5nYFOd7aZ9ooS65PRNyRbrlLxN1lSq219Z2h9MYIKLV3O8hE2GQ5+S9CBNeJ32nRAeWh6Jnwkg2KffDBNEH6MoE4c7EyIgyaXPLhKmVzYWJDR3zRhbXKWECuWJ1PEjSNr2MDthE3HFjJxKXNLlnfLc4t08JrQYmAZfKqW5yiPRPImLUnvUEbbUzE+mB+oJ0YHD2LqJ/FgZ0WtQz4SNPxD0T58Upqic0nTQdGR1kOBJ+7MQgowttyIiSL0wg7HCBnpou4HzdJh0Gb7MoBcB3eyby0DeB861xuUk6MgdxfBkdua6Fvg6+j1Bq7pgA8HBhQnaKLjIy0rCwzbhOdBwLhHrOlRJhouJct/NzRJiAOK++IZfi7CQVTsxBSc98A2W4smoY6m4l5YiOjtYpPG7IBGkbLW87V+JKA3lK9MMEZyg9E1vdcWxfcaoox6Vj1wnK3S4/2F1tOQncQFD+MME5h8SSYKQUVZ0F+pHr1DTdFAnxn9NvygSXYUg+6DZdaBmJ5ALoRoQJzcNloaQAH7lqwMTRDCJikByHgyV5M71tzcghGs20E49DVlqRolnJpQ5ABWmb3M1tznM4kHVZEnJBGTl6S4/w9XKX/NCck5/RbytzLQisKDZbnRBakma2LTp3sE3KB62FPKsINYc8WisjL9G4yEQgiYjt1HX9WOltnnat7ntmRGqTrAp6NKMNlojUEVNxOLpk5OoREcWXklDrIKHYDjtfKjXOKwPgdFxgeVp1conigKQo5ALH5o5EXXb+ibj65H2hOUzI5Kincv53cV040H/M9X+fL17LaP3F8yXux9c519P7nlp4/YPeFoLLzRy4PG14+XLl54mXj853nLMZf65ovxd+C/4+LdR+ZLr+MahfO0+eAAI/n9NC4eCOFStWrFixYsWKFStWrFixYsWKFStWrFixYsWKFSv+j+F/AJmJVJcPr9UyAAAAAElFTkSuQmCC" data-canonical-src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQkAAAC+CAMAAAARDgovAAAAkFBMVEX///8XNmAUNF8NMFwuR2wXOGJid5MvSW9DXH5KX3/4+foRM14mQmju8fRyg5xabouKmK2BkKXR1+CZpLVeco9CW37Iz9hofJfi5uva3+Xp7PAALFrx8/YxTnTAx9Lh5eqirr+4wMyvuMYAKFh6iqCVorNRaYmPnK+gq7s5VXkdP2nDytNSa4tsgJqqtcVod5GbDTtHAAATxUlEQVR4nO1dC3uyvLKlUECC3BFIRG6FcLHg//93JxFrUTGA9tvf2eewnud920oIw0oyk5lMIsetWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixYoVK1asWLFixf8TAAA0RKGFAPyrYoBeDLBMDBAyMaMuoMV2Wpmtu9sdetSKlDupB7V5krwvwqUeZEdZ6Su7uhdjp0hWldrBzBoyi4lo6uEwcnxFFgUKVVB7kN/3QrORyhRq0y+QMiUwj7PeA9lp+X0Q92c5foSgv8tYd45xOF1Du1dZMFn3hnaX4y1PHvcxAp4Qs8VWBieaRLMEpgjOdJOGdpZvGvK8D/5BDPKpeDiVEZqqxN9/8Hf4/eDjg8EESi1Mnv747Bs21MYoPWaLaJb6MfLsqwiTTIDIwltVZQiiCh+HtosnmWDhKROoO8nMp1/JEMRdHjHehjLBgDjBBIjyeqI9ejK2hsPkgs2E+IQJEPnsVriV4uNgPRfiPSagdeCZ91/Bq1sjZdT1Up8IrO1sGs5Q1V36TzAB0o06jwcKXm2s4E+ZsHf7+Y+/QOCfCfEGE4ElCoukUIXNU1O0fHSEmbzs8T14wbVH3+l1Juzv/aKeeZaiyZ7Ut7hPoEJe3CF6qMao4nyZiSNmyz4OQU7G5zhLmdBeJoI0CB6j4lUmIuU1QdRDNWrVFzIROp8vE0GoUEZG6YtMRHjpyPiBeujGqlzGBOhe7xEUwgb+EROe8ioRlIoxQ7ZMY0aHt4ggCuv7YZC+xISN3xFE2Nlv9ol484rVGILf5/cv9goTaELsKez9RzdkERP6u0QQKrbd+0wA572uScQoH2pdMjqixeZ7BOrGe5uJtHmXCbV5iDcs6BOh8X6XIBDzW1WxnAn49iAlqsK4N6ULmMi2f9AlSHPUt5p7MROAfcM88OL9KJ3PhHb6AwGoDGp+o64WM3Gs/0IQ1bhTmvOZeNeC/sog33SKCSb4eybCdpkf/Kzee9XdztWYoSVOVn6JMk1B8Idu6dI+0W3/pkVU99Y5nt0noMEWgFfF5vMgH7bNx1TIQN0OO8VCJpA+TQQNLQtTcSxVzl5jImUPDnXrmkmWpmlWWadaZEsr6APzsZCJCTlo7eLmpOutMRlNsm6M2Nz5BKgaVkFeSYIfiUEQlZvno4SGvIfDYxkTKJ94PV7FlYfCUIOZzp52qMrNnHtunwiZ8qqH2/hH6OnNqMSqwMtumQ4j/8uYmDIcPK9f3w9VNatbEJ05rHouE8GJNZtR3fuILSrFByHI2JWlyrtbhlnEBCgm9LaaDyQJsx2rB92a87lM2My4CJ8/eDTAuqmZF/ZfB7+zH5fmFjGBJvS2YNy4/ewogqoMC8/VEx6zV6qnxyg+8H96Ea+q/FYxvfGo2SImoseediOHfOdOAJPRh/gb52Nun/CYIRp+OxIRizf0FlUVP5U2eYzQvMLEXUd7lMO6ZxtuGLXvh3XPZeIos5f9DsXD+idIGkFtatdMmUujS5jQdmxrUD8Go1jTj/3Qms9lIppwv1TRcKK7lXEo7XQniidXiOczYY9bpB/w/uOaSscYHupmoN7m6onoc2oarYqyYTmpja6SA3vwx58wkTClJWP0sXrICEHz28Gg/Rs90YNYSaITTlYVzclXeIUJiSmtWnuP1YesYAY/GE2zmZjpCvOqIDYHpS3SYLo7LGUiZDeHaowtOEoMJvaDTvQ384k7ULPZ1JJznEzfWMaEzbahvD5Gfs54w731e8dcPcGeY46JRTOMtmRKOZVttYCJlB1IFUfUBMc5rD7R/g7j2X6H+Up8RBW+xB2xoiw2FjBRfLFK8o9RWoqO4Z4L+NfazWWC+KIvRoqI4tjkjGyr+UwAna0w5dHpW3rbJ3iep8GLPodrX/+O39nxieiN6OE5tWc8Z2ARE5g5QtV6VC1F+3Ma3p6CGLemkeVDvVOwdNJzy5k7OgbRu+CtgC6vNrgYn3HPZyKcmGHiUdvtyYeN4kq6ZTpORYNJR8+2IYRBgDTthdk2B5xXh0cPXhVxNibqfCZidrhKMEaZBuSVz5nDIWBl7YL5se1FdnQMvPqZjxj8+UxMzO4E6elrzsF8JoD1dnBd3eN3sgYmfJ99/g8ycbMuCuu3F+H4vfJAxXwmMmYolUyT/kEmbtfKnfeW6ntxlXvfYD4TLLeSVs1Mrf5TJjTlD7IGVHyXxTGfiYRd9VfxH2OCi97LLbq8mn6rNv87mLjLLgLVxJLOHPBNcWNM5zPhsFX2f7JPcGE5M0maBfVw4x/MZ6Jg1/seEwvmE73Y5tvpLNTwDyPh/519gshdvpOQ2YO/jdLO1xP/IhMjedtaghekz49D2A1mFQuYYFvRr/IfZGI0l9/Ot9M7S9gQkleY+N8zn7jKnknNe1wI+Hd4/DfOMX8RJ4b4Dhe8+Du9+ju/Q3+PCea08dnuJw6gzOW/Xp9yDrTbfCZs9sRu77/HxIs74ggZ0KxF4UXtKXy/wETADps9iU/MxNL5xN3dnokPIv9CrFeVr6HUP4tZDZXPEKUitZZZdZdgFULnkM1D0OYtJghQVJ522w9BnZN0N3zBq6JYEMf8nohjju40A8aXcIHYyIcaY9fXc7NMsjQ6DoKrr+2SvCXD60zChrof30s8Cn5/TXxbEOWfiG1/juxVuC4HnhuKv26zpns8RbF5Ico/jhD1S8Bh4KWFjun+8plD5XcdbgETFXu9Yzu63oFG8zd7KQW8fOXnHiCMvcyxTvrVhQAIelnp181kJuRZhuuEcAETEdtYic6YpB5rDUyfOzpGmQAhTBPLx/Vn8yGIN7kbgNBxNHEzPU6E6zRoARMxM7zO8w+7aCgKFhPly3pCs9OiVeqt+KHSkcZ/PO7hIYU8azs11RCu06AFTICJ7NjRtXLWG+67eeU+fvsEaWz7WJGur3597Yc7ynlxdFMubCeyBV9igrnu/RD46BEzpmN8MwiqzmECRp31fWj2X/sRDSDc7w7ogdqJdMFXRsfkYrn5uLRUMcrf2N0ZTACdHvXx1Cao5uiEZiJfULiOuyVMsBqYinIfLSa1u4w7BGl2xlmvJ0z2hGabjFERTDgJr1jRyQ1YTXnfKTpWfthQYc4aHd5EOujo9mTIdBx54WpzFuVjTiy53OePc7bBStxuhoZvDhPhjt0S6jb3HroFO+jIvzLb5ibd0Q91c6M0oc9S3Dc7moE058Qea2J7pPpRW7dcIOfAHlI3Hhir3W7PqUHt1ME49WCsehJrVYIX9KGyt+oNC/1otrfNdgKN7FYRPHt5IYxKPFFc/PWgNZNdeXVDcSZPCdK0l0aJnZpds3yj4ArDZaGfeQCz1afR5pZZFkVpWvlkcel35xHI2EVvz5YJymlR8rJKksqxpsqZN2n3/tVlHcXF1qWNOAd8P6Sol8fGh/g7QjWLVf5+pwtIPqdr793NCTHuZx/z5phIEh5OmmLhPAwZl4c+4M15ViM13e0Dgy6j+MfEg29kuEt3n+d3gOy91KJ7qMmvBAv3i4Lkb7Yyf/B35w3MGx1cIO2Z5ZZhf7vyw666uDPQgfE3onzebpLkyjkakyDFzHLLgIdKO0yYZaWHs4ZStsjzIBn3qc2zrChtuZJZbhF2N0dbhcWOicdDZZzdH8iA73N7/LlnIXrK8MhC9vGFbAi3m3eXn4UYbNhCz8Fjetr8fMxkcq/LTE0lmjdbg5busCdI387tEZSHoI6/Z1galR/ErEJzkSF9jrt9hKRPMEUYO50l2bIs7zSEkdOL8q3MwjAlITipzLLz0Nwn34Um+4ZqxOcPyQz9DRm2Y5UWrsTCzYiGzKLz4Lb355yFCfuO0XPqApMt94QQzkicTd9+snCbpnLcsUtPY3u/tZV6YOxKx/oEca/yN4TQxyO/zAF1f7JXvVffGJ7qvnkcn9QrZ+HJaW9IF14TRRXa0WNLF6532MbC9c8hePExzjgVnxi1HWeA15Lf1OZhn/EPEyz67vsEoaJtXu0V6qc+tsODzidYIjw9FRIU8nJR1G355OBra6Ow8LjHLDbrmnnLM9RKNboeEDrs+0YPLuypyPBSUQ7KaPiZosQGC/eHPlHJo5J5yxNIzmOss6+vYt/47IxTCrtqFwnhFyN7ay9oBUYo48kKcVDsVH4qXHIbOhHd9NneUTrHZMVTmGcra6krsm4fgudxxtjA+tJaObCtw/xkCV4QlerlU6af64leFFjsZomiCgfz2aa815mg579b8tccCYjlFL4z1s7qV23HlYvYbKZE4YUvVWfysMAXfQDqXJpj9fw1aFRT3CrlxMH8xHYwndEZ5/KjxJXp4v3Isfw/J6I82Zg4wER8YjQ344q404162wdy+Z+JBjVV6pmEA9Yd9pn8FGHBjiQ8tx1DQPqVEc1Qkosc4nYnlccZZx8ENhPPjyzvAZDXlflJqQ/ytjkHwBviGx1qxdXNJJr6moZZIszZpU8RwrTIT3h3+GyoJFSOWjF0M7OnvznjjwDC2D6mWVJVRVkWFd2d6tlzv8bkjyUJbC/NqsopCrpLNvLgjKNA/gE5fvAvPHtUkH9bjhUr/o8A2SCwQ/LD9jyi5uh/iPwfAwCJ4ocBmcPZxD4QQKLEgZ2S6QkilhGRq5qnkcu9dQnPVziY0gNZ+mJ2ENJKe5CSRIWfP0D95YA+iqOTRITIL+H54V58LgEh/TymNZ0f6tGHEpBC5PO+IHd+mkbzpclE/iwueQKKIhTS6/SFyEcBfQt6E6kD9q9BPowiIvW5rl+z0m3saptxFc4VWcnbrWvkmYR9pQukjQ2kioslJT66texauOOqjYsrkKkVl1qIq8SKC6SzbxdWWKo7zjYMLCHQKYbSabvK3mxc1zfknUGupUbARRvFkNIOu7uKK2TfaMkbB99JdrCAjUvjsDOKA7klanV6/KNJCK8TQIpvCtDJ2PXtg4Ti3PKJOMQv84g0lgapD2ajkwK57yjWiasXSRsZm987HZuocjXOysOMPLKwsKzo1kGLW2xIkNsYMWp/EweTnV01uua4QaZkgVXHMbKNCpXY9ptKcx3Ow3UUxpZhezhBcoEqDLutgdIchT7WucA9B9egW6DOCc0WHus02JhBIcHasYm7EQcQW9DyY59M0qJdFsf2d4WKmj7iqBBXN8BVVtfQxp59KqF3SIJY81sak6JMHBJoOEGKgwRHQRDIShq3UWBh0pjAl1DkRhW2Y2zCtkk4nEb4GJvHuMJR7OooU46OQZmArhl0tRd9d8iRUWXYtlGCXd3FfnTDhOF6jsQRUTizjhGARgcKbOtuC0mf6HS/AFx5CqGR2FuPNMQx+faz1NJsXGF0YSLwfQ+FgVFxoZTbdcrFaXBmIiASfxeETwuT/hxtUg1EG1LLwS6M2MYXJiSjsL/jMO84eEiQBlr/l4mj4nHACwkTCKBdbnmtzTkSoMfqJVyY2pYOuPzk5ZIf4tQjs2oyg0gNCE46iH6YOCopp3UalI5cIiM/ByCXQqznxzsmLNMsJSIsYWLb5jY0jFxJgtbRM/KaVldJYc9EF31CQlia+GnrWFrmB/h4YQJE7ka3YyXjOF2KdjQIEBImNthPAGUCWA3N0452rt51mNRy8ApZl3w6OnCStQlOjTjUEw42uO209rdPVCmGsEpQIru6hxTyZMqEG9KjD2hMFOkWaSj3mHd+aqSaszEcrWdiZxkW6plI6coCURguZSJwTcCZCjKSvLhjovR8ye+ZqInigUae1wi1aSYpCcJWXkPOPDNx3NqcraSJpEmGFeRGVzuxdAk9o1QxPNrGun+sj/Qb8OrKVhyIzn2Cy2T7zER1hJlin5nYFOd7aZ9ooS65PRNyRbrlLxN1lSq219Z2h9MYIKLV3O8hE2GQ5+S9CBNeJ32nRAeWh6Jnwkg2KffDBNEH6MoE4c7EyIgyaXPLhKmVzYWJDR3zRhbXKWECuWJ1PEjSNr2MDthE3HFjJxKXNLlnfLc4t08JrQYmAZfKqW5yiPRPImLUnvUEbbUzE+mB+oJ0YHD2LqJ/FgZ0WtQz4SNPxD0T58Upqic0nTQdGR1kOBJ+7MQgowttyIiSL0wg7HCBnpou4HzdJh0Gb7MoBcB3eyby0DeB861xuUk6MgdxfBkdua6Fvg6+j1Bq7pgA8HBhQnaKLjIy0rCwzbhOdBwLhHrOlRJhouJct/NzRJiAOK++IZfi7CQVTsxBSc98A2W4smoY6m4l5YiOjtYpPG7IBGkbLW87V+JKA3lK9MMEZyg9E1vdcWxfcaoox6Vj1wnK3S4/2F1tOQncQFD+MME5h8SSYKQUVZ0F+pHr1DTdFAnxn9NvygSXYUg+6DZdaBmJ5ALoRoQJzcNloaQAH7lqwMTRDCJikByHgyV5M71tzcghGs20E49DVlqRolnJpQ5ABWmb3M1tznM4kHVZEnJBGTl6S4/w9XKX/NCck5/RbytzLQisKDZbnRBakma2LTp3sE3KB62FPKsINYc8WisjL9G4yEQgiYjt1HX9WOltnnat7ntmRGqTrAp6NKMNlojUEVNxOLpk5OoREcWXklDrIKHYDjtfKjXOKwPgdFxgeVp1conigKQo5ALH5o5EXXb+ibj65H2hOUzI5Kincv53cV040H/M9X+fL17LaP3F8yXux9c519P7nlp4/YPeFoLLzRy4PG14+XLl54mXj853nLMZf65ovxd+C/4+LdR+ZLr+MahfO0+eAAI/n9NC4eCOFStWrFixYsWKFStWrFixYsWKFStWrFixYsWKFSv+j+F/AJmJVJcPr9UyAAAAAElFTkSuQmCC" width="400" height="400" />
# 
# <!-- <font size="2"> -->
#     
# Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.
# 
# Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma. Existing AI approaches have not adequately considered this clinical frame of reference. Dermatologists could enhance their diagnostic accuracy if detection algorithms take into account “contextual” images within the same patient to determine which images represent a melanoma. If successful, classifiers would be more accurate and could better support dermatological clinic work.
# 
# In this competition, you’ll identify melanoma in images of skin lesions. In particular, you’ll use images within the same patient and determine which are likely to represent a melanoma. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.
# 
# Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.
# 
# <!-- </font>  -->
# *Notes: This is from https://www.kaggle.com/c/siim-isic-melanoma-classification/ description.*

# ---
# 
# ### Clarification with multiple versions
# 
# With these multiple versions - I'm trying to change some stuff in the notebook, some tuning, some new augmentation or something but most of the time I get some Kaggle Notebooks errors. 
# 
# So, I have to rerun it a lot because the errors are sometime random and I need to circumvent it.
# 
# ---

# <font size="+3" color="green"><b>My other works</b></font><br>
# 
# <div class="row">
#   <div class="col-sm-4">
#     <div class="card">
#       <div class="card-body" style="width: 20rem;" style='background:red'>
#         <h5 class="card-title"><u>Transformer-XL Intro and Baseline</u></h5>
#         <img style='height:200px' src="data:image/png;base64,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" class="card-img-top" alt="..."><br>
#         <p class="card-text">A guide to the transformer architecture and its application in m5.</p>
#         <a href="https://www.kaggle.com/nxrprime/transformer-xl-intro-baseline" class="btn btn-primary" style="color:white;">Go to Post</a>
#       </div>
#     </div>
#   </div>
#     
#   <div class="col-sm-4">
#     <div class="card">
#       <div class="card-body" style="width: 20rem;">
#          <h5 class="card-title"><u>Cornell Birdcall EDA</u></h5>
#          <img style='height:200px' src="https://storage.googleapis.com/kaggle-competitions/kaggle/19596/logos/thumb76_76.png?t=2020-05-22-02-14-05" class="card-img-top" alt="..."><br>
#          <p class="card-text">An EDA for the cornell birdcall competition</p>
#          <a href="https://www.kaggle.com/nxrprime/simple-eda-preprocess-and-geographical-viz" class="btn btn-primary" style="color:white;">Go to Post</a>
#       </div>
#     </div>
#   </div>    

# ## Some preprocessing methods are from Neuron Engineer and Ben Graham - credit where credit is due.

# ## What is melanoma?

# Let's watch a video!

# In[1]:


from IPython.display import YouTubeVideo,HTML
YouTubeVideo("mkYBxfKDyv0", width=800, height=500)


# We learnt that:
# + Melanoma destroys melanocytes i.e skin cells.
# + It has a number of symptoms like:
#     + Bleeding
#     + Patchy skin
#     + Light eye color
#     etc.

# In[2]:


import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
from plotly import tools
import os
import seaborn as sns
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
from plotly.tools import FigureFactory as FF
import plotly
from tqdm import tqdm
import cv2
from PIL import Image
from plotly.offline import iplot
import cufflinks
import cv2 as cv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
get_ipython().run_line_magic('load_ext', 'autoreload')
plotly.io.templates.default = "none"
import altair as alt; import altair_render_script
plt.style.use("fivethirtyeight")
def _generate_bar_plot_hor(df, col, title, color, w=None, h=None, lm=0, limit=100):
    cnt_srs = df[col].value_counts()[:limit]
    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], orientation = 'h',
        marker=dict(color=color))

    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    annotations = []
    annotations += [go.layout.Annotation(x=673, y=100, xref="x", yref="y", text="(Most Popular)", showarrow=False, arrowhead=7, ax=0, ay=-40)]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


# In[3]:


# Images Example
train_images_dir = '../input/siim-isic-melanoma-classification/train/'
train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]
test_images_dir = '../input/siim-isic-melanoma-classification/test/'
test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
print('5 Training images', train_images[:5]) # Print the first 5


# In[4]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


# 

# In[5]:


train.head()


# # SIIM-ISIC: EDA and Augmentation

# Hello all and welcome. In this notebook I shall give you an overview of the new SIIM-ISIC Melanoma Detection competition and also a brief overview into our data augmentation techniques using the python library `albumentations`. If you like, do remember to upvote as that's where I get my motivation from.

# Our model needs to diagnose:
# + Target
# + Benign/Malignant
# + Diagnosis

# We will also try some preprocessing ideas such as:
# + **Reducing lighting-condition effects** : as we will see, images come with many different lighting conditions, some images are very dark and difficult to visualize. We can try to convert the image to gray scale, and visualize better. Alternatively, there is a better approach. We can try the method of Ben Graham,
# + **Cropping uninformative area** : everyone know this :) Here, I just find the codes from internet and choose the best one for you :)

# **<h1 id="one" style="color:purple;">1. Introduction </h1>**

# Let's look at first 20 files:

# In[6]:


fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 4
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images_dir + train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(-ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


# In[7]:


train.head()


# From this we can infer a LOT of domain knowledge such as:
# + The `diagnosis` feature is a simple diagnosis of the cancer.
# + The `benign_malignant	` feature is a feature that determines whether the tumor is benign or malignant (benign is harmless, malignant is harmful)
# + The `anatom_site_general_challenge` tells us where the cancer is.
# + The `target` is the target feature.

# Now for missing values:

# In[8]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[9]:


# Missing values statistics
missing_values = missing_values_table(train)
missing_values.head(20)


# Now we can see the percentages of missing values.

# Hmm... so both `age_approx` and `sex` have 0.2% percent missing values.

# In[10]:


missing_values = missing_values_table(test)
missing_values.head(20)


# So we can see that the percentage of missing values **doubles** in `anatom_site_general_challenge` from train to test. Back to our general EDA now.

# This is simple enough if one bothers to take a look - one patient has more than 114 growths. The patients growths could potientially be similar, so we can observe the similarities between the growths.

# In[11]:


train['age_approx2'] = train['age_approx'].fillna(0).astype('str')


# We can use the magic of **d3.js** to view categorical variables at a glance.

# In[12]:


holiday_json = {
"name": "flare",
"children": [
{
"name": "Affected Part",
"children":[
{
    "name": "Torso",
         "children": [
              {"name": "Chest", "size": 3.0},
              {"name": "Stomach (outer covering)", "size": 1.0},
              {"name": "Back of torso", "size": 2.0}
     ]
},
{
    "name": "Head/Neck",
         "children": [
              {"name": "Eyes", "size": 3.0},
              {"name": "Nose", "size": 1.0},
              {"name": "Neck", "size": 2.0},
              {"name": "Lips", "size": 1.0},
              {"name": "Ears", "size": 1.0}
     ]
},
{"name": "Upper extremity                                      ", "size": 3.0},
{"name": "Lower extremity                                      ", "size": 3.0},
{"name": "Palms/soles                                          ", "size": 2.0},
{"name": "Oral/genitalia                                       ", "size": 1.0}
]
},
{
"name":  "diagnosis",
"children":[
{"name": "unkown",       "size": 10.0},
{"name": "nevus",       "size": 4.0},
{"name": "melanoma",   "size": 3.0},
{"name": "seborheic keratosis",       "size": 2.0},
{"name": "lichenoid keratosis",         "size": 2.0},
{"name": "solar lentigo",                          "size": 1.0},
{"name": "atypical melanocytic proliferation",     "size": 1.0},
{"name": "cafe-au-lait macule", "size": 1.0}
]
},
{
"name": "Target feature",
"children": [
{"name": "malignant          ", "size": 1.0},
{"name": "benign                    ", "size": 6.0}
]
},
{
"name": "Age (approximated)",
"children":[
{
    "name": "0",
         "children": [
              {"name": "1", "size": 1.0},
              {"name": "2", "size": 1.0},
              {"name": "3", "size": 1.0},
              {"name": "4", "size": 1.0},
              {"name": "5", "size": 1.0}
     ]
},
{
    "name": "5",
         "children": [
              {"name": "6", "size": 1.0},
              {"name": "7", "size": 1.0},
              {"name": "8", "size": 1.0},
              {"name": "9", "size": 1.0},
              {"name": "10", "size": 1.0}
     ]
},
{
    "name": "20",
         "children": [
              {"name": "11", "size": 1.0},
              {"name": "12", "size": 1.0},
              {"name": "13", "size": 1.0},
              {"name": "14", "size": 1.0},
              {"name": "15", "size": 1.0},
              {"name": "16", "size": 1.0},
              {"name": "17", "size": 1.0},
              {"name": "18", "size": 2.0},
              {"name": "19", "size": 1.0},
              {"name": "20", "size": 1.0}
     ]
},
{
    "name": "25",
         "children": [
              {"name": "21", "size": 1.0},
              {"name": "22", "size": 1.0},
              {"name": "23", "size": 2.0},
              {"name": "24", "size": 1.0},
              {"name": "25", "size": 3.0}
     ]
},
{
    "name": "30",
         "children": [
              {"name": "26", "size": 2.0},
              {"name": "27", "size": 1.0},
              {"name": "28", "size": 2.0},
              {"name": "29", "size": 2.0},
              {"name": "30", "size": 1.0}
     ]
},
{
    "name": "40",
         "children": [
              {"name": "31", "size": 1.0},
              {"name": "32", "size": 3.0},
              {"name": "33", "size": 2.0},
              {"name": "34", "size": 1.0},
              {"name": "35", "size": 3.0},
              {"name": "36", "size": 2.0},
              {"name": "37", "size": 1.0},
              {"name": "38", "size": 2.0},
              {"name": "39", "size": 1.0},
              {"name": "40", "size": 4.0}
     ]
},
{
    "name": "45",
         "children": [
              {"name": "41", "size": 5.0},
              {"name": "42", "size": 4.0},
              {"name": "43", "size": 4.0},
              {"name": "44", "size": 5.0},
              {"name": "45", "size": 5.0}
     ]
},
{
    "name": "50",
         "children": [
              {"name": "46", "size": 4.0},
              {"name": "47", "size": 5.0},
              {"name": "48", "size": 4.0},
              {"name": "49", "size": 4.0},
              {"name": "50", "size": 3.0}
     ]
},
{
    "name": "55",
         "children": [
              {"name": "51", "size": 3.0},
              {"name": "52", "size": 3.0},
              {"name": "53", "size": 3.0},
              {"name": "54", "size": 2.0},
              {"name": "55", "size": 3.0}
     ]
},
{
    "name": "60",
         "children": [
              {"name": "56", "size": 3.0},
              {"name": "57", "size": 3.0},
              {"name": "58", "size": 2.0},
              {"name": "59", "size": 2.0},
              {"name": "60", "size": 4.0}
     ]
},
{
    "name": "65",
         "children": [
              {"name": "61", "size": 2.0},
              {"name": "62", "size": 3.0},
              {"name": "63", "size": 4.0},
              {"name": "64", "size": 3.0},
              {"name": "65", "size": 3.0}
     ]
},
{
    "name": "70",
         "children": [
              {"name": "66", "size": 3.0},
              {"name": "67", "size": 2.0},
              {"name": "68", "size": 2.0},
              {"name": "69", "size": 2.0},
              {"name": "70", "size": 2.0}
     ]
},
{
    "name": "75",
         "children": [
              {"name": "71", "size": 2.0},
              {"name": "72", "size": 2.0},
              {"name": "73", "size": 1.0},
              {"name": "74", "size": 1.0},
              {"name": "75", "size": 2.0}  
         ]
},
{
    "name": "80",
         "children": [
              {"name": "76", "size": 2.0},
              {"name": "77", "size": 2.0},
              {"name": "78", "size": 1.0},
              {"name": "79", "size": 1.0},
              {"name": "80", "size": 1.0}
     ]
},
{
    "name": "85",
         "children": [
              {"name": "81", "size": 2.0},
              {"name": "82", "size": 1.0},
              {"name": "83", "size": 1.0},
              {"name": "84", "size": 1.0},
              {"name": "85", "size": 1.0}
     ]
},
{
    "name": "90",
         "children": [
              {"name": "86", "size": 1.0},
              {"name": "87", "size": 1.0},
              {"name": "88", "size": 1.0},
              {"name": "89", "size": 1.0},
              {"name": "90", "size": 1.0},
     ]
},
]
}
] 
}               

import IPython
import json

with open('output.json', 'w') as outfile:  
    json.dump(holiday_json, outfile)
pd.read_json('output.json').head()

#Embedding the html string
html_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>

.node {
  cursor: pointer;
}

.node:hover {
  stroke: #000;
  stroke-width: 1.5px;
}

.node--leaf {
  fill: white;
}

.label {
  font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;
  text-anchor: middle;
  text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;
}

.label,
.node--root,
.node--leaf {
  pointer-events: none;
}

</style>
<svg width="760" height="760"></svg>
"""

js_string="""
 require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });

  require(["d3"], function(d3) {

   console.log(d3);

var svg = d3.select("svg"),
    margin = 20,
    diameter = +svg.attr("width"),
    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

var color = d3.scaleSequential(d3.interpolateViridis)
    .domain([-4, 4]);

var pack = d3.pack()
    .size([diameter - margin, diameter - margin])
    .padding(2);

d3.json("output.json", function(error, root) {
  if (error) throw error;

  root = d3.hierarchy(root)
      .sum(function(d) { return d.size; })
      .sort(function(a, b) { return b.value - a.value; });

  var focus = root,
      nodes = pack(root).descendants(),
      view;

  var circle = g.selectAll("circle")
    .data(nodes)
    .enter().append("circle")
      .attr("class", function(d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })
      .style("fill", function(d) { return d.children ? color(d.depth) : null; })
      .on("click", function(d) { if (focus !== d) zoom(d), d3.event.stopPropagation(); });

  var text = g.selectAll("text")
    .data(nodes)
    .enter().append("text")
      .attr("class", "label")
      .style("fill-opacity", function(d) { return d.parent === root ? 1 : 0; })
      .style("display", function(d) { return d.parent === root ? "inline" : "none"; })
      .text(function(d) { return d.data.name; });

  var node = g.selectAll("circle,text");

  svg
      .style("background", color(-1))
      .on("click", function() { zoom(root); });

  zoomTo([root.x, root.y, root.r * 2 + margin]);

  function zoom(d) {
    var focus0 = focus; focus = d;

    var transition = d3.transition()
        .duration(d3.event.altKey ? 7500 : 750)
        .tween("zoom", function(d) {
          var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
          return function(t) { zoomTo(i(t)); };
        });

    transition.selectAll("text")
      .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
        .style("fill-opacity", function(d) { return d.parent === focus ? 1 : 0; })
        .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
        .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });
  }

  function zoomTo(v) {
    var k = diameter / v[2]; view = v;
    node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
    circle.attr("r", function(d) { return d.r * k; });
  }
});
  });
 """
from IPython.core.display import display, HTML, Javascript
h2 = display(HTML("""<h2 style="font-family: 'Garamond';"> Zoomable Circle Packing </h2> <i>This is all primarily based on Anisotropic's work: upvote there, I admire his work a lot.</i>"""))
h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)


# We can see that in 'age', middle aged people are more likely to get melanoma (and cancer in general as studies have shown) than people before and past their prime. It is also observable that we have a class distribution in `target` with many more 0's than 1's (I am poor at d3.js so it may not look that way :-)).
# 
# We also are using the DICOM file type here. It would be interesting to read in a DICOM file and soak in all the metadata.

# Here we can see the trees in the data and the grouping - codes are modified from Shivam Bansal's work. Next!

# In[13]:


doc = {"name": "Characteristics", "color": "#ffae00", "percent": "", "value": "", "size": 25, "children": []}

def getsize(s):
    if s > 80:
        return 30
    elif s > 65:
        return 20
    elif s > 45:
        return 15
    elif s > 35:
        return 12
    elif s > 20:
        return 10 
    else:
        return 5
def vcs(col):
    vc = train[col].value_counts()
    keys = vc.index
    vals = vc.values 
    
    ddoc = {"name": col, "color": "#be5eff", "percent": "", "value": "", "size": 25, "children": []}
    for i,x in enumerate(keys):
        percent = round(100 * float(vals[i]) / sum(vals), 2)
        size = getsize(percent)
        collr = "#fc5858"
 
        doc = {"name": x+" ("+str(percent)+"%)", "color": collr, "percent": str(percent), "value": str(vals[i]), "size": size, "children": []}
        ddoc['children'].append(doc)
    return ddoc

# Coding Backgrounds
doc['children'].append(vcs('anatom_site_general_challenge'))
doc['children'].append(vcs('benign_malignant'))
doc['children'].append(vcs('diagnosis'))
doc['children'].append(vcs('sex'))

html_d1 = """<!DOCTYPE html><style>.node text {font: 12px sans-serif;}.link {fill: none;stroke: #ccc;stroke-width: 2px;}</style><svg id="four" width="760" height="900" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>"""
js_d1="""
require(["d3"], function(d3) {
var treeData = """ +json.dumps(doc) + """
var root, margin = {
        top: 20,
        right: 90,
        bottom: 120,
        left: 90
    },
    width = 960 - margin.left - margin.right,
    height = 660,
    svg = d3.select("#four").attr("width", width + margin.right + margin.left).attr("height", height + margin.top + margin.bottom).append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")"),
    i = 0,
    duration = 750,
    treemap = d3.tree().size([height, width]);

function collapse(t) {
    t.children && (t._children = t.children, t._children.forEach(collapse), t.children = null)
}

function update(n) {
    var t = treemap(root),
        r = t.descendants(),
        e = t.descendants().slice(1);
    r.forEach(function(t) {
        t.y = 180 * t.depth
    });
    var a = svg.selectAll("g.node").data(r, function(t) {
            return t.id || (t.id = ++i)
        }),
        o = a.enter().append("g").attr("class", "node").attr("transform", function(t) {
            return "translate(" + n.y0 + "," + n.x0 + ")"
        }).on("click", function(t) {
            t.children ? (t._children = t.children, t.children = null) : (t.children = t._children, t._children = null);
            update(t)
        });
    o.append("circle").attr("class", "node").attr("r", function(t) {
        return t.data.size
    }).style("fill", function(t) { return t.data.color;
    }), o.append("text").attr("dy", ".35em").attr("x", function( t) {
        return t.children || t._children ? -13 : 13
    }).attr("text-anchor", function(t) {
        return t.children || t._children ? "end" : "start"
    }).text(function(t) {
        return t.data.name
    });
    var c = o.merge(a);
    c.transition().duration(duration).attr("transform", function(t) {
        return "translate(" + t.y + "," + t.x + ")"
    }), c.select("circle.node").attr("r", function(t) {
        return t.data.size
    }).style("fill", function(t) {
        return t.data.color
    }).attr("cursor", "pointer");
    var l = a.exit().transition().duration(duration).attr("transform", function(t) {
        return "translate(" + n.y + "," + n.x + ")"
    }).remove();
    l.select("circle").attr("r", function(t) {
        return t.data.size
    }), l.select("text").style("fill-opacity", 1e-6);
    var d = svg.selectAll("path.link").data(e, function(t) {
        return t.id
    });
    console.log(), d.enter().insert("path", "g").attr("class", "link").attr("d", function(t) {
        var r = {
            x: n.x0,
            y: n.y0
        };
        return u(r, r)
    }).merge(d).transition().duration(duration).attr("d", function(t) {
        return u(t, t.parent)
    });
    d.exit().transition().duration(duration).attr("d", function(t) {
        var r = {
            x: n.x,
            y: n.y
        };
        return u(r, r)
    }).remove();

    function u(t, r) {
        var n = "M" + t.y + "," + t.x + "C" + (t.y + r.y) / 2 + "," + t.x + " " + (t.y + r.y) / 2 + "," + r.x + " " + r.y + "," + r.x;
        return console.log(n), n
    }
    r.forEach(function(t) {
        t.x0 = t.x, t.y0 = t.y
    })
}(root = d3.hierarchy(treeData, function(t) {
    return t.children
})).x0 = height / 2, root.y0 = 0, root.children.forEach(collapse), update(root);
});
"""
js7m="""require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });
 
 require(["d3"], function(d3) {// Dimensions of sunburst.
 
 


var svg = d3.select("#fd"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(120).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("fd.json", function(error, graph) {
  if (error) throw error;

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

// Define the div for the tooltip
var div = d3.select("body").append("div")	
    .attr("class", "tooltip")				
    .style("opacity", 0);

  var node = svg.append("g")
      .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", function(d) {return d.size})
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)).on("mouseover", function(d) {		
            div.transition()		
                .duration(200)		
                .style("opacity", .9);		
            div	.html(d.id )
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    
// node.append("title")
  //  .text(function(d) { return d.id; });

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);
      

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
});

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
 });
"""

h = display(HTML(html_d1))
j = IPython.display.Javascript(js_d1)
IPython.display.display_javascript(j)


# <h1 id="xxx" style="color:purple">2. A nice trick with 3-dimensional visualization</h1>

# <h2 id="#exp">Explanation</h2>
# 
# Quite simply, we are going to use 3-dimensional plotting to visualize the spatial restrictions of our image as well as the depth of the image and the 3-dimensional representation of the growth in space.

# <h2 id="app">Application</h2>

# In[14]:


ds = pydicom.dcmread(train_images_dir + train_images[0])
from skimage import measure
def plot_3d(image, threshold=-300):
    p = image.transpose(2,1,0)
    
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=1, allow_degenerate=True) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.title("3-dimensional representation of an image")
    plt.show()

plot_3d(ds.pixel_array,threshold=100)


# This looks rather,well,odd but it helps us to get a general **picture of the outlines of our image in 3 dimensional perspective.** It helps to understand the size with a 3-dimensional spatial representatiion and at the same time helps to understand the limits of our image in a 3 dimensional scale. It was originally used for Data Science Bowl 2017 by Guido Zuldhof, but I have adapted it for here (this competition).
# 
# A recommended approach would be to **resample** the images before plotting it like this. Here we can clearly see what is inside the image and where the isolated cells are (it forms a figure vaguely resemblant of a skull). We can also get a touchy-feely sense of this whole thing with Plot.ly (it is a real toll on my poor computer with 3 gigabyte RAM)

# Will add plotly visualizations later.

#  **<h1 id="two" style="color:purple;">3. Benign and malignant tumors</h1>**

# **What is a benign tumor?**
# 
# A benign tumor put simply is one that will not cause any cancerous growth. It will not damage anythin, it's just a small blot on the landscape of your skin.
# 
# **What is a malignant tumor?**
# 
# A malignant tumor is the evil twin of  the benign tumor: it causes cancerous growth.

# In[15]:


df = pd.DataFrame(train.benign_malignant.value_counts())
df['name'] = df.index
alt.Chart(df).mark_bar().encode(
    x='name',
    y='benign_malignant',
    tooltip=["name","benign_malignant"]
).interactive()


# As expected, we have a HUGE class distribution here - we will need to use undersampling or oversampling to deal with this.

# <h2 id="benign">Benign image viewing</h2>

# In[16]:


fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 5
for i in range(1, columns*rows +1):
    # added the grid lines for pixel purposes
    ds = pydicom.dcmread(train_images_dir + train[train['benign_malignant']=='benign']['image_name'][i] + '.dcm')
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


# These are the first 30 melanoma images with benign tumors. Let's check the distribution of values in `benign_malignant`.

# In[17]:


train.benign_malignant.value_counts()


# We have significantly more benign tumors than malignant ones. Why is it so? Well, medically, most of the time cancer patients only reach cancerous conditions with malignant tumors. Stage 3 and Stage 4, the most critical level conditions of cancer are only reached when the patient is in serious trouble. And, well, cancerous growth on cells is never a good thing.
# 
# Benign tumors are frequently the first signs of cancerous growth pacing the way for the malignant tumors. Now we can view malignant tumors:

# <h2 id="malignant">Malignant image viewing</h2>

# In[18]:


# BY SERGEI ISSAEV
vals = train[train['benign_malignant']=='malignant']['image_name'].index.values
fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 5
for i in range(1, columns*rows +1):
    # added the grid lines for pixel purposes

    ds = pydicom.dcmread(train_images_dir + train[train['benign_malignant']=='malignant']['image_name'][vals[i]] + '.dcm')
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


# **<h1 id="three" style="color:purple;"> 4. Which part of the body?</h1>**

# Melanoma or skin cancer occurs, well, wherever your skin is. Oof. That makes it sort of unavoidable really if you get it because heart and lung cancer are localized to just one area whereas melanoma can sprout literally anywhere.

# Let's see where the most frequent amounts of cancerous growth occur:

# In[19]:


df = pd.DataFrame(train.anatom_site_general_challenge.value_counts())
df['name'] = df.index
alt.Chart(df).mark_bar().encode(
    x='name',
    y='anatom_site_general_challenge',
    tooltip=["name","anatom_site_general_challenge"]
).interactive()


# It seems like we have a lot of issues with the torso, and after that the extremities of the body (upper/lower). We have about 100-200 cases of cancer in the mouth or genitalia (the areas with the lowest rate of cancerous growth) and the palms and soles also are safe (probably because they are not exposed to any external sources in the day-to-day life of the person). 
# 
# One could indeed fathom that the torso is frequently exposed either during the occasional workout or the occasional swim, or in some cases, the occasional extreme adventure. It also could be that the torso was exposed to UV light (which is the cause of melanoma) in highly populated regions (where pollution allows the sun's UV rays to come in).

# In[20]:


import squarify
fig = plt.figure(figsize=(25, 21))
marrimeko=train.anatom_site_general_challenge.value_counts().to_frame()
ax = fig.add_subplot(111, aspect="equal")
ax = squarify.plot(sizes=marrimeko['anatom_site_general_challenge'].values,label=marrimeko.index,
              color=sns.color_palette('cubehelix_r', 28), alpha=1)
ax.set_xticks([])
ax.set_yticks([])
fig=plt.gcf()
fig.set_size_inches(40,25)
plt.title("Treemap of cancer counts across different parts", fontsize=18)
plt.show();


# Our earlier hypothesis now has more depth to it, with half (approximately?) of the skin cancer cases located in the torso. However, the torso has more square area than any other affected part, so it seems like the area on a body part is correlated with number of cases on that body part (as we can see).

# In[21]:


_generate_bar_plot_hor(train, 'anatom_site_general_challenge', '<b>Affected Location</b>', '#66f992', 800, 400, 200)


# In[22]:


def view_images(images, title = '', aug = None):
    width = 6
    height = 4
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.show()


# In[23]:


get_ipython().run_cell_magic('time', '', 'for i in train[\'anatom_site_general_challenge\'].unique()[:4]:\n    view_images(train[train[\'anatom_site_general_challenge\']==i][\'image_name\'], title=f"Growth in the {i}")')


# We need to isolate malignant tumors from all this. So how do we do that? With the magic of dataframe indexing!

# In[24]:


get_ipython().run_cell_magic('time', '', 'for i in train[\'anatom_site_general_challenge\'][:1].unique():\n    view_images(train[train[\'anatom_site_general_challenge\']==i][train[\'target\']==1][\'image_name\'], title=f"Malignant in the {i}");')


# In each malignant tumor, we can observe that the tumor has a visceral orange halo in some cases and in other cases it does not. What I think this means is that the orange halo occurs when the patient has reached stage 3/stage 4 of serious cancerous growth. In the cases without a halo, I think that it means that the patient is having a malignant tumor but has not reached critical levels of cancer yet.
# 
# I would like to think of the orange halo as being an event horizon of some sort - a point of no return for a killed cell (unless chemotherapy of course). As grim as this may sound, remember that in this scenario we are supposed to be helping the doctors do their duty.

# **<h1 id="four"  style="color:purple;">5. Diagnosis</h1>**

# OK, so let's look at the diagnosis feature:

# In[25]:


_generate_bar_plot_hor(train, 'diagnosis', '<b>Affected Location</b>', '#f2b5bc', 800, 400, 200)


# We have seven types of diagnosed cancerous growths here:
# + Unkown: a possibly novel type of growth
# + Nevus: (from Google) a usually non-cancerous disorder of pigment-producing skin cells commonly called birth marks or moles.
# + Melanoma: Skin cancer's form (what we are working with)
# + Seborrheic keratosis: Brown, waxy and patchy growths that are not related to skin cancer.
# + Lentigo NOS: A type of skin cancer that starts from the outside of the skin and attacks by going inword.
# + Lichenoid keratosis: It is a thin pigmented sort of plaque, if you will.
# + Solar lentigo: Like lentigo but caused by UV rays from the sun (very common in Delhi)
# + cafe-au-lait macule: French for "coffee with milk". These are brownish spots also called "giraffe spots".
# + atypical melanocytic proliferation: Abnormal quantities of melanin appear on the skin.

# In[26]:


fig = plt.figure(figsize=(25, 21))
marrimeko=train.diagnosis.value_counts().to_frame()
ax = fig.add_subplot(111, aspect="equal")
ax = squarify.plot(sizes=marrimeko['diagnosis'].values,label=marrimeko.index,
              color=sns.color_palette('cubehelix_r', 28), alpha=1)
ax.set_xticks([])
ax.set_yticks([])
fig=plt.gcf()
fig.set_size_inches(40,25)
plt.title("Treemap of cancer counts across different ages", fontsize=18)
plt.show();


# **80 percent of all the cases here are unknown. Literally 80 percent.**

# We also need to analyze how different these sort of skin growths are. Are they all similar of different? Do they look the same or not? And do they have the all-too-well known visceral orange halo? We shall see.
# 
# I'll be excluding `Unknown` for the purpose of clarity.

# In[27]:


view_images(train[train['diagnosis']=='nevus']['image_name'], title="Nevus pigmentatious growth");


# From Nevus pigmentatious growth, we can observe that nevus also has an orange halo which is not so visceral and pronounced. This can be confusing as our model can mix up this orange halo and the other orange halo

# In[28]:


view_images(train[train['diagnosis']=='melanoma']['image_name'], title="Melanoma's growth");


# Now that is cancer. Melanoma in most cases displays the vicious and visceral orange halo as it devours alive other poor cells who where unlucky enough to cross its path. 

# In[29]:


train.diagnosis.value_counts()


# In[30]:


def view_images_sp(images, title = '', aug = None):
    width = 1
    height = 1
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        plt.imshow(image, cmap=plt.cm.bone) 
        axs.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.show()
view_images(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo's growth");


# Hmmm.... lentigo has not only the orange halo of death but also the green halo, which could be what distinguishes lentigo from melanoma (I'm not great at biology). This green halo could be a special characteristic of lentigo that could help our model to distinguish lentigo from other carcinogenic growths. 

# In[31]:


view_images(train[train['diagnosis']=='lichenoid keratosis']['image_name'], title="Lichenoid's growth");


# Lichenoid mimics the dangers of real carcinogenous growth with its halo(s). Lichenoid is very good at mimicking other forms of cancer.

# In[32]:


view_images(train[train['diagnosis']=='seborrheic keratosis']['image_name'], title="Lichenoid's growth");


# Seborrheic and lichenoid keratosis are both rather similar.

# In[33]:


view_images_sp(train[train['diagnosis']=='atypical melanocytic proliferation']['image_name'], title="Atypical melanocytic's growth");


# Atypical melanocytic proliferation combines the scary things about melanoma and lentigo and brings them in one place complete with both halos. 

# **<h1 id="five"  style="color:purple;"> 6. Age</h1>**

# Age is an important factor in carciongenous growth, because it helps you to understand who is more vulnerable at an early age and who is more vulnerable at later stages of their life.

# In[34]:


df = pd.DataFrame(train.age_approx.value_counts())
df['name'] = df.index
alt.Chart(df).mark_bar().encode(
    x='name',
    y='age_approx',
    tooltip=["name","age_approx"]
).interactive()


# So we have a bell (Gaussian or normal distribution) of train data. What about test?

# In[35]:


fig= plt.figure(figsize=(22,6))
test["age_approx"].value_counts(normalize=True).to_frame().iplot(kind='bar',
                                                      yTitle='Percentage', 
                                                      linecolor='black', 
                                                      opacity=0.7,
                                                      color='red',
                                                      theme='pearl',
                                                      bargap=0.8,
                                                      gridcolor='white',                                                     
                                                      title='It does not exactly follow the same distribution in test though.')
plt.show()


# Middle-aged people are the most likely to get cancer whereas those on the RHS and LHS of the plot are least likely to get it.  We now have found out that there is a bell curve distribution for age.

# **<h1 id="ca" style="color:purple;"> 7. Image Preprocessing</h1>**

# **Why is preprocessing any different from augmentation?**
# Preprocessing is basically a filter - augmentation is basically adding more variations of the same image.

# <h2 id="norm">Normalization</h2>

# We use normalization to evenly distribute something across an image - we can use normalization to, for example, normalize lighting conditions across the image.

# In[36]:


import cv2
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = cv2.resize(image, (256, 256))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(256,256))
        image[:, :, 0] = clahe.apply(image[:, :, 0])
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# **<h1 id="six" style="color:purple;"> 8. Image Augmentations</h1>**

# We need to use image augmentations to add to our existing set of data. Why? To help our model identify the tumors correctly, even in grayscale colormap.

# In[37]:


import matplotlib
matplotlib.font_manager._rebuild()
with plt.xkcd():

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-30, 10])

    data = np.ones(100)
    data[70:] -= np.arange(30)

    ax.annotate(
        'THE DAY I REALIZED\nI COULD USE APTOS METHODS\nIN OTHER COMPS',
        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

    ax.plot(data)

    ax.set_xlabel('time')
    ax.set_ylabel('my overall sanity')
    fig.text(
        0.5, 0.05,
        '"My Mental Sanity Degrading over Time',
        ha='center')


# <h2 id="gray">Grayscale images</h2>

# We will first try to visualize in grayscale (only gray colors) so that it is possible for us to clearly visualize the varied differences in color, region, and shape.

# In[38]:


import cv2
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)  
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# <h2 id="ben">Ben Graham's method from 1st competition</h2>

# The first method we will use is **Ben Graham's method** from the first diabetic retionpathy detection competition.. it involves utilizing grayscale images as well as a gaussian blur afterwards.

# <h3 id="graytrain">Train</h3>

# In[39]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (256, 256))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)    
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Now we try it on the test set:

# <h3 id="graytest">Test</h3>

# In[40]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(test_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (256, 256))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)    
    plt.suptitle(title)
view_images_aug(test[test['sex']=='male']['image_name'], title="Images for Male");


# Nice! But we can further observe the clear color distinctions by using Neuron Engineer's method (an improved version of Ben Graham's).

# <h2 id="neuron">Neuron Engineer's method</h2>

# In[41]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)    
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Here we can finally visualize the clear distinctions in our data. The clear regions, the clear color differences, the clear everything! This is probably the best preprocessing method that we can apply... maybe not?

# <h2 id="circ">Circle crop</h2>

# In[42]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= circle_crop(image)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Circular crop has successfully worked, although it may not be feasible for images where the tumor is on the edge of the image. It does not seem so feasible, so I would recommend you try to be smarter in your methods for preprocessing. Remember, you can build upon Ben Graham's work as a starting point, then try Neuron Engineer's or circle crop or even build you own method.
# 
# 

# <h2 id="autocrop">Auto cropping</h2>

# Now we are using **auto-cropping** as a method of preprocessing, which is a more "refined" circle crop if you will. Think of circle crop as C, and think of auto-cropping as C++. Auto-cropping indeed is powerful, but the risk is that you will lose valuable data in the image.

# In[43]:


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= crop_image_from_gray(image)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# <h2 id="bgsub">Background subtraction</h2>

# Another thing you can do is **background subtraction.**

# In[44]:


fgbg = cv.createBackgroundSubtractorMOG2()
    
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= fgbg.apply(image)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# Oh god... looks like *the Conjuring* now. 

# <h2 id="seg">Image segmentation</h2>

# In[45]:


fgbg = cv.createBackgroundSubtractorMOG2()
    
def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        i = im // width
        j = im % width
        axs[i,j].imshow(thresh, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# This is called **image segmentation**. It breals down an image into its constituent parts represented by the distinction between regions (i.e in this instance the growth is white and the halo/surrrounding area is blackened). It helps our model to visually understand the distinctions even better than using grayscale images and it also helps to let our model identify the tumor.
# 
# However, it has a heavy downside as we lose all information inside and outside the growth and thus our model loses the capability to understand or learn something from the image.

# <h2 id="segf">A finer method for image segmentation</h2>

# In[46]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        i = im // width
        j = im % width
        axs[i,j].imshow(sure_bg, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# This is a finer form of image segmentation where we use a second threshold to finetune our segmented data in a sort of way. This helps us see some parts of the growth and also it lets us see the images and the halos in small amounts.

# <h2 id="segfg">Grayscale image segmentation</h2>

# In[47]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# These are now the segmented grayscale images, complete with markers. It will be a bit difficult for the model to learn anything from this due to the complete and utter confusion (pardon me) in the image with regard to the clear distinctions between image segments and there is no disctinction between parts of the image.

# <h2 id="fourier">Fourier method for pixel distributions</h2>

# Now we move on to something more interesting: **Fourier transforms.** 

# In[48]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        i = im // width
        j = im % width
        axs[i,j].imshow(magnitude_spectrum, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# What comes out is the magnitude spectrum of the image. It is helpful to understand where the majority of the growth is concentrated.

# <h2 id="albumentation">Albumentations library demonstration</h2>

# Also, you can use the albumentations library to create a lot of simulated images for your model. Remember, your model **MUST BE ABLE TO GENERALIZE!**

# In[49]:


import albumentations as A
image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
albumentation_list = [A.RandomSunFlare(p=1), A.RandomFog(p=1), A.RandomBrightness(p=1),
                      A.RandomCrop(p=1,height = 512, width = 512), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","RandomFog","RandomBrightness",
               "RandomCrop","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# We have much more augmentations we can try like:

# In[50]:


image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
albumentation_list = [A.RandomSunFlare(p=1), A.GaussNoise(p=1), A.CLAHE(p=1),
                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",
               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# Mess around with `p`:

# In[51]:


image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),
                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",
               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# Not done yet! We still have a few more tricks, namely erosion and dilation.

# <h2 id="erosion">Erosion</h2>

# In[52]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        kernel = np.ones((5,5), np.uint8) 
  
        # The first parameter is the original image, 
        # kernel is the matrix with which image is  
        # convolved and third parameter is the number  
        # of iterations, which will determine how much  
        # you want to erode/dilate a given image.  
        img_erosion = cv2.erode(image, kernel, iterations=1) 
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");


# <h2 id="dilation">Dilation</h2>
# 
# There! Now we can try dilation:

# In[53]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        kernel = np.ones((5,5), np.uint8) 
    
        img_erosion = cv2.dilate(image, kernel, iterations=1) 
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");


# Now for both!

# <h2 id="combo">Combination of erosion and dilation</h2>

# In[54]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        kernel = np.ones((5,5), np.uint8) 
  
        img_erosion = cv2.erode(image, kernel, iterations=1) 
        img_erosion = cv2.dilate(image, kernel, iterations=1) 
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");


# <h2 id="microscope">Roman's microscope augmentation</h2>

# **Original**:
# ```python
# class Microscope:
#     def __init__(self, p: float = 0.5):
#         self.p = p
# 
#     def __call__(self, img):
#         if random.random() < self.p:
#             circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
#                         (img.shape[0]//2, img.shape[1]//2),
#                         random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
#                         (0, 0, 0),
#                         -1)
# 
#             mask = circle - 255
#             img = np.multiply(img, mask)
# 
#         return img
# 
#     def __repr__(self):
#         return f'{self.__class__.__name__}(p={self.p})'
# ```

# <h2 id="put">Albumentations + erosion</h2>

# In[55]:


image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
image = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
kernel = np.ones((5,5), np.uint8) 
  
image = cv2.erode(image, kernel, iterations=1) 
albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),
                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",
               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# <h2 id="put2">Albumentations + dilation</h2>

# In[56]:


image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
image = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
kernel = np.ones((5,5), np.uint8) 
  
image = cv2.dilate(image, kernel, iterations=1) 
albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),
                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",
               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# <h2 id="put3">Albumentations + erosion + dilation</h2>

# In[57]:


image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))
image = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
kernel = np.ones((5,5), np.uint8) 
  
image = cv2.dilate(image, kernel, iterations=1) 
image = cv2.erode(image, kernel, iterations=1) 

albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),
                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",
               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]

##reminder of helper function
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")


# <h2 id="cw">Complex wavelet transform</h2>

# ### Horizontal detail

# In[58]:


import pywt
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= circle_crop(image)
        coeffs = pywt.dwt2(image, 'bior1.3')
        IM1, (IM2, IM3, IM4) = coeffs
        image = IM2
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# ### Vertical detail

# In[59]:


import pywt
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= circle_crop(image)
        coeffs = pywt.dwt2(image, 'bior1.3')
        IM1, (IM2, IM3, IM4) = coeffs
        image = IM3
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.gray) 
        axs[i,j].axis('off')
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# ## Hough transform

# In[60]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= circle_crop(image)
        circles = cv2.HoughCircles(imagee,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

        image = np.uint16(np.around(circles))
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.gray) 
        axs[i,j].axis('off')
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# ## Canny edges

# In[61]:


def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image= circle_crop(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)

        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

        image = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.gray) 
        axs[i,j].axis('off')
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");


# **<h1 id="et" style="color:purple;">9. 3-dimensional augmentation techniques</h1>**

# As you have seen <a href="#xxx">here, it is possible for one to make use of 3-dimensional images in this competition.</a> As such, if anyone is bold or foolhardy enough to try these, I shall make a guide for you to use these. (My poor, poor RAM).
# 
# It will be coming shortly as soon as I figure out how to properly work with MONAI.
# 
# ---

# **<h1 id="seven" style="color:purple;">10. Baseline Modeling</h1>**

# While everyone is trying to use EfficientNet and transfer learning, I think that **it is indeed great, but we need to try to IMPLEMENT ideas on our own.** So here, I am going to use `Depthwise Spatial Convolutions` instead of normal Conv2D layers.

# I will demonstrate 2 methods:
# + Implementation from grassroots (our own model)
# + Keras pretrained models.
# 
# **EDIT: As of 19/06/2020 I have removed grassroots modeling section. You are free to go back to older versions and check the sections on grassroots modeling over there.**

# In[62]:


X_train, X_val = train_test_split(train, test_size=0.2, random_state=42)
X_train['image_name'] = X_train['image_name'].apply(lambda x: x + '.jpg')
X_val['image_name'] = X_val['image_name'].apply(lambda x: x + '.jpg')
test['image_name'] = test['image_name'].apply(lambda x: x + '.jpg')
X_train['target'] = X_train['target'].apply(lambda x: str(x))
X_val['target'] = X_val['target'].apply(lambda x: str(x))


# In[63]:


from keras.applications import ResNet50 as model
from PIL import Image
model = model(weights='imagenet')
model.compile(
    'Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)


# It looks rather gargantuan but fundamentally it is just an overglorified convolutional neural network with some bells and whistles (and a lot of layers at that too)

# ### K-Fold Cross Validation

# In[64]:


print(f"Train set is {train.shape[0] / test.shape[0]} times bigger than test set.")


# So, as it seems like the train set is 3 times bigger than the test set, it makes sense for us to use a 3-fold cross validation split using sklearn's RepeatedKFold class. The data is split into $fold$ chunks (where $fold$ is the n_splits parameter of our model) and the training data is then passed to the model.

# Let's define a simple KFold cross validation that we should be able to use in some way.

# In[65]:


from sklearn.model_selection import RepeatedKFold
splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)


# Now the splitter sort of splits the data into chunks by adding a certain "feature" to the data which determines which batch/fold the data should go in. Here we have 3 batches / 3 folds where the data can be separated to.

# In[66]:


partitions = []

for train_idx, test_idx in splitter.split(train.index.values):
    partition = {}
    partition["train"] = train.image_name.values[train_idx]
    partition["validation"] = train.image_name.values[test_idx]
    partitions.append(partition)
    print("TRAIN:", train_idx, "TEST:", test_idx)
    print("TRAIN:", len(train_idx), "TEST:", len(test_idx))


# ### The basic structure of model
# 
# So, to think about the model we also have to define a preprocessing pipeline which is a very traumatic experience if you are using TensorFlow like I am. To put it into perspective, PyTorch data loaders are much more fast than TF dataloaders and the only reason we are using TF here is because of keras.applications.
# + **Let's use only simple augmentations.** Nothing too complex, just our circle crop augmentation will be enough to replicate the effect on some other images. Circle crop was first formulated by Tom Aindow and exemplified by Neuron Engineer during the APTOS competition (see their effect).
# + **TF dataloaders**. TF Dataloaders are very slow but they are effective in a way. I will use TensorFlow dataloaders over here to run the model. 
# + **One class for configuration and hyperparameters.** I will be using just one simple class Config for defining all the configuration and hyperparameters for accessibility.

# In[67]:


class Config:
    BATCH_SIZE = 8
    EPOCHS = 40
    WARMUP_EPOCHS = 2
    LEARNING_RATE = 1e-4
    WARMUP_LEARNING_RATE = 1e-3
    HEIGHT = 224
    WIDTH = 224
    CANAL = 3
    N_CLASSES = train['target'].nunique()
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5


# In[68]:


train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                 rotation_range=360,
                                 horizontal_flip=True,
                                 vertical_flip=True)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=X_train,
    directory='../input/siim-isic-melanoma-classification/jpeg/train/',
    x_col="image_name",
    y_col="target",
    class_mode="raw",
    batch_size=Config.BATCH_SIZE,
    target_size=(Config.HEIGHT, Config.WIDTH),
    seed=0)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

valid_generator=validation_datagen.flow_from_dataframe(
    dataframe=X_val,
    directory='../input/siim-isic-melanoma-classification/jpeg/train/',
    x_col="image_name",
    y_col="target",
    class_mode="raw", 
    batch_size=Config.BATCH_SIZE,   
    target_size=(Config.HEIGHT, Config.WIDTH),
    seed=0)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  
        dataframe=test,
        directory = '../input/siim-isic-melanoma-classification/jpeg/test/',
        x_col="image_name",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        target_size=(Config.HEIGHT, Config.WIDTH),
        seed=0)


# Now we can train the model. (***NOTE: I HAVE COMMENTED THIS BECAUSE IT TAKES A LOT OF TIME)***

# In[69]:


"""
%%time
STEP_SIZE_TRAIN = train_generator.n//64
STEP_SIZE_VALID = valid_generator.n//64
history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    validation_data=valid_generator,
                                    validation_steps=STEP_SIZE_VALID,
                                    epochs=1,
                                    verbose=1).history
"""                                    


# I have trained the model for only a few epochs because of time constraints.

# **<h1 id="eight" style="color:purple;"> 11. CutMix, MixUp, AugMix and GridMask</h1>**

# This is probably the most famous augmentation method in recent history. CutMix, MixUp and AugMix are all very popular nowadays, so I have given it a go here (using TPUs).

# Due to RAM Constraints, I will display only **the images** of CutMix and MixUp:

# ## CutMix
# 
# ![](https://www.kaggleusercontent.com/kf/35223031/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..-mWTqxqQM59nQBlLwUBoPw.A6K69GsBx6Ja5zzjt5dN3v6Us2yn43u2vSH0FIsonWndRhMLQse4ijiQh9VtCRGoBWu0BswTdZm7r0E267Cm8do_7SeQloSx0KFqtR2OgdwgNAC-kH4w4Zk2tsAN_Qcy7DUfDjEp43YIoPeR8hsUZkUoG0oNH9E1H95ALe0KhJSB8skpVqy9wZlSKuWTFkTikFw_i9UcLt_3k7QMs3gDXWvt3o1GjZJVcmn59WcoVR0JJMrMf9PEzYGX0B1V-HfaliEzNsi8sG7_cSkaAill2psFaVg-caMOdZ0hJ3dtKBypHGQVNqRGfD3c0f2plHjoARA6h6O5DrCzh4XajLQMmtphXTFTNSv69R91d3dsMdu6nXoPcnegiAbo0Fvf0rsAPAFTEpEtqKHRqGxbt5uMNqkmR84Mtl7LUtWtdUMGjM6YmQRe4emxWvqpl9LKvB3jQWK4Iu-Rg1yDyFQSLHAOGh2Apf-9OM_4ZO8P22-PNWX4qxchy6T9-UO5qtLZD6-0hbi9vpcHcwChta409fCDx4SILAEe8st2thLr6400Ko0WYQh4E0XAooOoW74kcaqznkcvMW_HMoUxftXSYHVK8iTL4_Mz3J7JSiPYcRHD7eSbsHXjODp7WD06yT0oUm5iQmBRs2doio8GoLpzESZbAXjlvsSRJjdsB1k1oWQg1-u_YwPXo1hnVSJYuOnjj2NC.jdCubfO-gaFURT7SF4sIqQ/__results___files/__results___90_0.png)

# ## MixUp
# 
# ![](https://www.kaggleusercontent.com/kf/35223031/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..-mWTqxqQM59nQBlLwUBoPw.A6K69GsBx6Ja5zzjt5dN3v6Us2yn43u2vSH0FIsonWndRhMLQse4ijiQh9VtCRGoBWu0BswTdZm7r0E267Cm8do_7SeQloSx0KFqtR2OgdwgNAC-kH4w4Zk2tsAN_Qcy7DUfDjEp43YIoPeR8hsUZkUoG0oNH9E1H95ALe0KhJSB8skpVqy9wZlSKuWTFkTikFw_i9UcLt_3k7QMs3gDXWvt3o1GjZJVcmn59WcoVR0JJMrMf9PEzYGX0B1V-HfaliEzNsi8sG7_cSkaAill2psFaVg-caMOdZ0hJ3dtKBypHGQVNqRGfD3c0f2plHjoARA6h6O5DrCzh4XajLQMmtphXTFTNSv69R91d3dsMdu6nXoPcnegiAbo0Fvf0rsAPAFTEpEtqKHRqGxbt5uMNqkmR84Mtl7LUtWtdUMGjM6YmQRe4emxWvqpl9LKvB3jQWK4Iu-Rg1yDyFQSLHAOGh2Apf-9OM_4ZO8P22-PNWX4qxchy6T9-UO5qtLZD6-0hbi9vpcHcwChta409fCDx4SILAEe8st2thLr6400Ko0WYQh4E0XAooOoW74kcaqznkcvMW_HMoUxftXSYHVK8iTL4_Mz3J7JSiPYcRHD7eSbsHXjODp7WD06yT0oUm5iQmBRs2doio8GoLpzESZbAXjlvsSRJjdsB1k1oWQg1-u_YwPXo1hnVSJYuOnjj2NC.jdCubfO-gaFURT7SF4sIqQ/__results___files/__results___92_0.png)

# Due to RAM constraints this is how GridMask looks like:-
# ![](https://www.kaggleusercontent.com/kf/35223031/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..63LkdMRmfzzZaD0K20kflQ.rCbILKa3mSHswXZAKu4cxN_7nrFcFMyhAIb2nLfaMaqv61Q0ByPYzoIwVFVlTt5Awre_NajFwfjRGCU7FoMUqOTKcRP1uNl4-myKV4tiTYakMhArR849e4LZmLYITaxX-MTDqEHtp4HHQwVewgcf4h1ocIQZ5BEe4IbgcKqItjS3a27djq8TdOvMr693UEAXTx3vNjBZy5-AjzbqPDKpJysg_FzDeOEqS_c0FYZb4O2KksjgL9eq1u4hzw6itrFXhArvXLaAo-QQDH7oK58YvS1h9jTsrKtk6AzMyDrBhyYCTSNhlP2Uqq__3LkjtcB0DpHPx2mdkf7JbNx8W_2C6JCsL7zv0XEuoBhNBNNmXSPyFmn0med1CgEMEV4oF2vgQ4Cnjm8EVllfchq1U0E73W_9NYW7WCAP76rNI5VsrbCa0l_g4sPzCJEiBd2WH9bOv3NefF_rogNO0Pa4mTYgsuUyE9XgbFriIfceZZEov4Zae4jSLiLe4cmrfmmiGKUUb94hMb-dPSkHkfL2nL8ab9zs1rGbmtUcnKlJGiU3witwH9kr-B2ydvLUephf1QU9JTAfLb5DzmFnDyGDlOh_Um1Q0VOeFCKoWXASgYeKJQ__HmaYdFT3AE0RNZiPj-yAN0XedIv5nlwBr8APoT_aQ-SeA0nS_2aI1oZeUQvxflhclzMV-UUBerPjQ2O-iDJE.9o0Do1qM183_XG_hCkXq-A/__results___files/__results___96_0.png)

# ---
# 
# **<h1 id="nine" style="color:purple;">12. More Data</h1>**

# ---
# 
# We have two additional datasets that we can use:
# + [Skin Cancer MNIST](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
# + [Skin Lesion Images for Melanoma Classification](https://www.kaggle.com/andrewmvd/isic-2019)
# 
# ---
# 
# I got this from **Alex Shonenkov's brilliant kernel: https://www.kaggle.com/shonenkov/merge-external-data/ Please upvote his work if you like it.**

# <h1 id="ten" style="color:purple"> 13. Attack the model</h1>

# How do we think that a model can classify something correctly? How do we measure whether small disturbances in image like adding a noise or a random flip or something can affect whether an image is benign or malignant? Could we measure peturbance - or rather how much the model has been affected - in any way?
# 
# Attacking a machine learning model or adversarial learning is by no means a new technique - it's been around for a while. There were NIPS challenges and Karen Fink/Allunia's notebook on how to attack a machine learning model. Here we have to try and implement these ideas.

# A simple enough attack would be to slightly alter an image by adding some noise.

# In[70]:


import random
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = sp_noise(image, 0.192)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");


# This is a very basic way to add noise - it's known as salt and pepper noise. Here however we can implement multiple noises from a glance.

# In[71]:


import random
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def view_images_aug(images, title = '', aug = None):
    width = 6
    height = 5
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    for im in range(0, height * width):  
        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))
        image = data.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = sp_noise(image, 0.192)
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
view_images_aug(train[train['diagnosis']=='unknown']['image_name'], title="Lentigo NOS's Erosion");


# ---
# 
# <h1 id="a"> <b>APPENDIX A: Melanoma</b></h1>
# 
# ---

# In[72]:


YouTubeVideo("bQLphecl-1A", height=500, width=800)


# In[73]:


YouTubeVideo("=-O3fLMg6qwQ", height=500, width=800)


# ---
# 
# <h1 id="b"><b>APPENDIX B: ISIC Overview</b></h1>
# 
# ---

# *Source: https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/isicArchive*
# 
# The International Skin Imaging Collaboration: Melanoma Project is an academia and industry partnership designed to facilitate the application of digital skin imaging to help reduce melanoma mortality. When recognized and treated in its earliest stages, melanoma is readily curable. Digital images of skin lesions can be used to educate professionals and the public in melanoma recognition as well as directly aid in the diagnosis of melanoma through teledermatology, clinical decision support, and automated diagnosis. 
# 
# Currently, a lack of standards for dermatologic imaging undermines the quality and usefulness of skin lesion imaging. ISIC is developing proposed standards to address the technologies, techniques, and terminology used in skin imaging with special attention to the issues of privacy and interoperability (i.e., the ability to share images across technology and clinical platforms). In addition, ISIC has developed and is expanding an open source public access archive of skin images to test and validate the proposed standards. This archive serves as a public resource of images for teaching and for the development and testing of automated diagnostic systems.

# ---
# 
# <h1 id="c"><b>APPENDIX C: ISIC Winner Solutions</b></h1>
# 
# ---

# + 1st place: https://isic-challenge-stade.s3.amazonaws.com/99bdfa5c-4b6b-4c3c-94c0-f614e6a05bc4/method_description.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=E3f4iijeBEYmUWuhJUJtFLi%2Fx%2Fw%3D&Expires=1591213432
# + 2nd place: http://www.skinreader.cn/
# + 

# <h2 style="color:red;">WORK IN PROGRESS</h2>

# ---
# 
# <h2 style="color:red;"> If you liked it, please upvote!</h2>
