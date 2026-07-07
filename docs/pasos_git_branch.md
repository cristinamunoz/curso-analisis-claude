# Guía: crear tu branch personal y subir tu trabajo a GitHub

Esta guía es para quienes nunca usaron git ni una terminal. La
sigues **una sola vez**, antes de empezar la Mañana 2, para dejar
tu branch personal lista. Ver también
[pasos_curso_manana2.md](pasos_curso_manana2.md).

Cada paso muestra el comando exacto. Recuerda la regla del curso:
**nunca ejecutes un comando sin entender qué hace** — si tienes
dudas, pregunta a los instructores antes de continuar.

---

### Paso 0 · Requisitos

- Tener `git` y `gh` (GitHub CLI) instalados. Para comprobarlo:

  ```
  git --version
  gh --version
  ```

  Si algún comando falla, avisa a los instructores antes de seguir.

- Tener una cuenta de GitHub y haber aceptado la invitación como
  colaborador del repositorio del curso (revisa tu correo o
  pregunta a los instructores si no te llegó).

### Paso 1 · Autenticarte con GitHub

Solo se hace una vez por computador:

```
gh auth login
```

Sigue las preguntas en pantalla: elige **GitHub.com**, **HTTPS**, y
autentícate en el navegador que se abrirá automáticamente. Cuando
termine, confirma que quedó bien con:

```
gh auth status
```

Debería decir "Logged in" con tu usuario de GitHub.

### Paso 2 · Clonar el repositorio del curso

Si todavía no tienes el repositorio en tu computador:

```
gh repo clone cristinamunoz/curso-analisis-claude
cd curso-analisis-claude
```

Si ya lo tienes (por ejemplo, de la Mañana 1), solo entra a la
carpeta y actualízalo:

```
cd curso-analisis-claude
git checkout main
git pull
```

### Paso 3 · Crear tu branch personal

Elige un nombre único, en minúsculas y con guiones — por ejemplo,
si te llamas María Pérez: `maria-perez`. Todo tu trabajo del curso
irá en esta branch.

```
git checkout -b tu-nombre-apellido
```

Verifica que quedaste parada/o en tu nueva branch:

```
git branch
```

Debería aparecer tu branch marcada con un asterisco (`*`).

### Paso 4 · Primer push (crear la branch en GitHub)

Aunque todavía no hayas cambiado nada, sube tu branch vacía ahora
para reservar tu espacio en GitHub y comprobar que el acceso
funciona:

```
git push -u origin tu-nombre-apellido
```

La opción `-u` (de "upstream") solo es necesaria esta primera vez:
le dice a git que, de ahora en adelante, tu branch local queda
conectada con la branch del mismo nombre en GitHub, así los
próximos `git push` no necesitan más argumentos.

Ve a la página del repositorio en GitHub y confirma que tu branch
aparece en el listado de branches.

### Paso 5 · Trabajar y guardar tu progreso durante el curso

Durante la Mañana 2 vas a crear/editar archivos (tu `CLAUDE.md`
personal, `solution.py`, figuras en `outputs/`, etc.). Cada vez que
quieras guardar un avance:

```
git add nombre-del-archivo
git commit -m "Descripción breve de lo que cambié"
git push
```

- `git add` selecciona qué archivos vas a guardar en este punto.
- `git commit` guarda ese punto con un mensaje explicando qué
  hiciste (en español, breve y claro).
- `git push` sube ese guardado a GitHub, a tu branch.

No hace falta hacer esto después de cada línea de código — alcanza
con un commit por cada resultado relevante (por ejemplo: "agrego
cálculo de diversidad alfa", "exporto figura de correlación con
aridez").

### Paso 6 · Verificar que todo quedó subido

Al final de la Mañana 2, confirma que no te quedó nada sin subir:

```
git status
```

Si dice "nothing to commit, working tree clean" y "Your branch is
up to date with 'origin/tu-nombre-apellido'", tu trabajo quedó
guardado correctamente en GitHub.

---

**Resultado esperado:** tienes una branch personal en GitHub
(`tu-nombre-apellido`), conectada a tu carpeta local, lista para
recibir tus commits durante la Mañana 2.
