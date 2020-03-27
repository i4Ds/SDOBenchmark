# SDOBenchmark

## Training and Test Results

DATASET: 

* IMG_SIZE = 50
  * BATCH_SIZE = 100
    * EPOCHS = 1
      * ~0.5
    * EPOCHS = 5
      * ~0.69 - 0.79
* IMG_SIZE = 256
  * BATCH_SIZE = 100
    * EPOCHS = 5
      * ~0.515



| DATASET                | IMG_SIZE | BATCH_SIZE | EPOCHS | TEST_AMOUNT | RESULT           |
| ---------------------- | -------- | ---------- | ------ | ----------- | ---------------- |
| 682xflare_682xno_flare | 50       | 100        | 1      | 0.1         | ~0.5             |
| 682xflare_682xno_flare | 50       | 100        | **5**  | 0.1         | **~0.69 - 0.79** |
| 682xflare_682xno_flare | 256      | 100        | 5      | 0.1         | ~0.515           |
| 682xflare_682xno_flare | 256      | 100        | 1      | 0.1         | ~0.515           |
| 682xflare_682xno_flare | 80       | 100        | 1      | 0.1         | ~0.522           |
| 682xflare_682xno_flare | 80       | 100        | 5      | 0.1         | ~0.706-0.757     |
|                        |          |            |        |             |                  |
|                        |          |            |        |             |                  |
|                        |          |            |        |             |                  |

