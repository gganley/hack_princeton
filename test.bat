for /r %%i in (*) do pandoc %%i -o %%i.pdf -V title: -V geometry:margin=0in
