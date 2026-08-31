#include <Servo.h>

// ============================================================
// PIN DEFINITIONS
// ============================================================

// X axis
#define X1 3
#define X2 4
#define X3 5
#define X4 6

// Y axis
#define Y1 7
#define Y2 8
#define Y3 9
#define Y4 10

// Pen servo
#define SERVO_PIN 11


// ============================================================
// SETTINGS
// ============================================================

#define PEN_UP_ANGLE 120
#define PEN_DOWN_ANGLE 150

// Smaller = faster
#define STEP_DELAY 2


// ============================================================
// STEPPER SEQUENCE
// ============================================================

const byte stepSequence[8][4] = {
  {1, 0, 0, 0},
  {1, 1, 0, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 0},
  {0, 0, 1, 1},
  {0, 0, 0, 1},
  {1, 0, 0, 1}
};


// ============================================================
// VARIABLES
// ============================================================

int xStepIndex = 0;
int yStepIndex = 0;

long currentX = 0;
long currentY = 0;

Servo pen;


// ============================================================
// MOTOR CONTROL
// ============================================================

void setXMotor(byte a, byte b, byte c, byte d)
{
  digitalWrite(X1, a);
  digitalWrite(X2, b);
  digitalWrite(X3, c);
  digitalWrite(X4, d);
}


void setYMotor(byte a, byte b, byte c, byte d)
{
  digitalWrite(Y1, a);
  digitalWrite(Y2, b);
  digitalWrite(Y3, c);
  digitalWrite(Y4, d);
}


// ============================================================
// X STEP
// ============================================================

void stepX(int direction)
{
  xStepIndex += direction;

  if (xStepIndex > 7)
    xStepIndex = 0;

  if (xStepIndex < 0)
    xStepIndex = 7;

  setXMotor(
    stepSequence[xStepIndex][0],
    stepSequence[xStepIndex][1],
    stepSequence[xStepIndex][2],
    stepSequence[xStepIndex][3]
  );

  delay(STEP_DELAY);
}


// ============================================================
// Y STEP
// ============================================================

void stepY(int direction)
{
  yStepIndex += direction;

  if (yStepIndex > 7)
    yStepIndex = 0;

  if (yStepIndex < 0)
    yStepIndex = 7;

  setYMotor(
    stepSequence[yStepIndex][0],
    stepSequence[yStepIndex][1],
    stepSequence[yStepIndex][2],
    stepSequence[yStepIndex][3]
  );

  delay(STEP_DELAY);
}


// ============================================================
// RELEASE MOTORS
// ============================================================

void releaseMotors()
{
  digitalWrite(X1, LOW);
  digitalWrite(X2, LOW);
  digitalWrite(X3, LOW);
  digitalWrite(X4, LOW);

  digitalWrite(Y1, LOW);
  digitalWrite(Y2, LOW);
  digitalWrite(Y3, LOW);
  digitalWrite(Y4, LOW);
}


// ============================================================
// PEN
// ============================================================

void penUp()
{
  pen.write(PEN_UP_ANGLE);
  delay(250);
}


void penDown()
{
  pen.write(PEN_DOWN_ANGLE);
  delay(250);
}


// ============================================================
// MOVE X/Y TO ABSOLUTE POSITION
// ============================================================

void moveTo(long targetX, long targetY)
{
  long dx = targetX - currentX;
  long dy = targetY - currentY;

  int xDirection;

  if (dx >= 0)
    xDirection = 1;
  else
    xDirection = -1;

  int yDirection;

  if (dy >= 0)
    yDirection = 1;
  else
    yDirection = -1;

  dx = abs(dx);
  dy = abs(dy);

  long totalSteps = max(dx, dy);

  if (totalSteps == 0)
    return;

  float xAccumulator = 0;
  float yAccumulator = 0;

  for (long i = 0; i < totalSteps; i++)
  {
    xAccumulator += (float)dx / totalSteps;
    yAccumulator += (float)dy / totalSteps;

    if (xAccumulator >= 1.0)
    {
      stepX(xDirection);

      xAccumulator -= 1.0;

      currentX += xDirection;
    }

    if (yAccumulator >= 1.0)
    {
      stepY(yDirection);

      yAccumulator -= 1.0;

      currentY += yDirection;
    }
  }

  releaseMotors();
}


// ============================================================
// COMMAND PROCESSING
// ============================================================

void processCommand(String command)
{
  command.trim();

  // ----------------------------------------------------------
  // PEN UP
  // ----------------------------------------------------------

  if (command == "PENUP")
  {
    penUp();

    Serial.println("OK");

    return;
  }


  // ----------------------------------------------------------
  // PEN DOWN
  // ----------------------------------------------------------

  if (command == "PENDOWN")
  {
    penDown();

    Serial.println("OK");

    return;
  }


  // ----------------------------------------------------------
  // HOME
  // ----------------------------------------------------------

  if (command == "HOME")
  {
    penUp();

    moveTo(0, 0);

    Serial.println("OK");

    return;
  }


  // ----------------------------------------------------------
  // MOVEMENT
  //
  // Example:
  //
  // M X100 Y200
  // ----------------------------------------------------------

  if (command.startsWith("M"))
  {
    int xIndex = command.indexOf('X');
    int yIndex = command.indexOf('Y');

    if (xIndex >= 0 && yIndex >= 0)
    {
      int xEnd = command.indexOf(' ', xIndex);

      String xString;

      if (xEnd >= 0)
      {
        xString = command.substring(
          xIndex + 1,
          xEnd
        );
      }
      else
      {
        xString = command.substring(
          xIndex + 1,
          yIndex
        );
      }

      String yString = command.substring(
        yIndex + 1
      );

      long targetX = xString.toInt();
      long targetY = yString.toInt();

      moveTo(
        targetX,
        targetY
      );

      Serial.println("OK");

      return;
    }
  }


  // ----------------------------------------------------------
  // UNKNOWN COMMAND
  // ----------------------------------------------------------

  Serial.println("ERROR");
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  // X pins
  pinMode(X1, OUTPUT);
  pinMode(X2, OUTPUT);
  pinMode(X3, OUTPUT);
  pinMode(X4, OUTPUT);

  // Y pins
  pinMode(Y1, OUTPUT);
  pinMode(Y2, OUTPUT);
  pinMode(Y3, OUTPUT);
  pinMode(Y4, OUTPUT);

  // Servo
  pen.attach(SERVO_PIN);

  // Start with pen up
  penUp();

  // Motors off
  releaseMotors();

  // Serial
  Serial.begin(115200);

  delay(1000);

  Serial.println("PLOTTER READY");
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  if (Serial.available())
  {
    String command = Serial.readStringUntil('\n');

    processCommand(command);
  }
}